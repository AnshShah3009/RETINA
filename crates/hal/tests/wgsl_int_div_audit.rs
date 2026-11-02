//! Keeps `docs/wgsl_int_div_audit.md` honest.
//!
//! naga wraps every integer `/` and `%` in a polyfill that replaces a zero
//! divisor (and `INT_MIN / -1`) with 1, because SPIR-V and MSL leave those
//! cases undefined. wgpu 30 can turn the polyfill off per shader module
//! (`ShaderRuntimeChecks::int_div_checks`), which is only sound when no shader
//! can ever divide by zero. The audit document classifies every integer
//! division in the shader corpus; this test re-derives the list from naga's IR
//! and fails when the two disagree, so a new or changed `/` or `%` cannot land
//! without its divisor being argued for.
//!
//! The scan is exact rather than textual: each source is parsed and validated
//! by naga, and only `Binary { Divide | Modulo }` expressions whose result type
//! is a signed or unsigned integer (scalar or vector) are counted. Float
//! division, constant-folded expressions and comments never show up. It needs
//! no adapter, so it runs in CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use wgpu::naga;

/// Path of the audit document, relative to the workspace root.
const AUDIT_DOC: &str = "docs/wgsl_int_div_audit.md";

/// Directories (relative to the workspace root) whose `.wgsl` files are
/// compiled by the workspace's compute paths.
const WGSL_DIRS: &[&str] = &[
    "crates/hal/shaders",
    "crates/hal/src",
    "crates/calib3d/src",
    "crates/features/src",
];

/// Directories whose `.rs` files may embed WGSL in raw string literals.
const INLINE_DIRS: &[&str] = &["crates/hal/src"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/hal sits two levels below the workspace root")
        .to_path_buf()
}

fn collect(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

/// Every raw string literal in `text`, with the identifier it is bound to
/// (`let NAME = r#"..."#` or `fn NAME() -> &'static str { r#"..."# }`).
fn raw_string_literals(text: &str) -> Vec<(String, String)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 1 < b.len() {
        let preceded_by_ident = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if b[i] == b'r' && b[i + 1] == b'#' && !preceded_by_ident {
            let mut j = i + 1;
            while j < b.len() && b[j] == b'#' {
                j += 1;
            }
            if j < b.len() && b[j] == b'"' {
                let terminator = format!("\"{}", "#".repeat(j - i - 1));
                let body_start = j + 1;
                if let Some(rel_end) = text[body_start..].find(&terminator) {
                    let name = binding_name(&text[..i]);
                    out.push((name, text[body_start..body_start + rel_end].to_string()));
                    i = body_start + rel_end + terminator.len();
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// The name of the `let` or `fn` closest before a literal.
fn binding_name(before: &str) -> String {
    let ident_after = |at: usize, kw: &str| -> String {
        before[at + kw.len()..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect()
    };
    let let_at = before.rfind("let ");
    let fn_at = before.rfind("fn ");
    match (let_at, fn_at) {
        (Some(l), Some(f)) if l > f => ident_after(l, "let "),
        (_, Some(f)) => ident_after(f, "fn "),
        (Some(l), None) => ident_after(l, "let "),
        (None, None) => "unnamed".to_string(),
    }
}

/// `(source id, WGSL text)` for the whole corpus. File shaders are keyed by
/// their workspace-relative path; inline shaders by `path#binding_name`.
fn corpus() -> Vec<(String, String)> {
    let root = workspace_root();
    let rel = |p: &Path| p.strip_prefix(&root).unwrap().display().to_string();
    let mut out = Vec::new();

    let mut files = Vec::new();
    for dir in WGSL_DIRS {
        collect(&root.join(dir), "wgsl", &mut files);
    }
    files.sort();
    files.dedup();
    for f in files {
        out.push((rel(&f), std::fs::read_to_string(&f).unwrap()));
    }

    let mut rs = Vec::new();
    for dir in INLINE_DIRS {
        collect(&root.join(dir), "rs", &mut rs);
    }
    rs.sort();
    for f in rs {
        let text = std::fs::read_to_string(&f).unwrap();
        for (name, body) in raw_string_literals(&text) {
            if body.contains("@compute") {
                out.push((format!("{}#{name}", rel(&f)), body));
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Sign {
    Unsigned,
    Signed,
}

/// One integer division or modulo found in the IR.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Site {
    source: String,
    op: &'static str,
    divisor: String,
    sign: Sign,
    /// The divisor folded to constants, if it is one (`None` = runtime value).
    constant: Option<Vec<i64>>,
    line: usize,
}

/// Fold `expr` to the integer constant(s) it denotes, if it is constant.
fn const_value(
    module: &naga::Module,
    exprs: &naga::Arena<naga::Expression>,
    expr: naga::Handle<naga::Expression>,
) -> Option<Vec<i64>> {
    fn literal(l: &naga::Literal) -> Option<i64> {
        match *l {
            naga::Literal::U32(v) => Some(v as i64),
            naga::Literal::I32(v) => Some(v as i64),
            naga::Literal::U64(v) => i64::try_from(v).ok(),
            naga::Literal::I64(v) => Some(v),
            naga::Literal::AbstractInt(v) => Some(v),
            _ => None,
        }
    }
    match exprs[expr] {
        naga::Expression::Literal(ref l) => literal(l).map(|v| vec![v]),
        naga::Expression::ZeroValue(_) => Some(vec![0]),
        naga::Expression::Constant(c) => {
            const_value(module, &module.global_expressions, module.constants[c].init)
        }
        naga::Expression::Splat { value, .. } => const_value(module, exprs, value),
        naga::Expression::Compose { ref components, .. } => {
            let mut all = Vec::new();
            for &c in components {
                all.extend(const_value(module, exprs, c)?);
            }
            Some(all)
        }
        _ => None,
    }
}

/// Parse, validate and scan one source. `Err` carries naga's diagnostic.
fn scan(source: &str, wgsl: &str) -> Result<Vec<Site>, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(wgsl))?;

    let mut functions: Vec<(&naga::Function, &naga::valid::FunctionInfo)> = module
        .functions
        .iter()
        .map(|(h, f)| (f, &info[h]))
        .collect();
    for (i, ep) in module.entry_points.iter().enumerate() {
        functions.push((&ep.function, info.get_entry_point(i)));
    }

    let mut sites = Vec::new();
    for (func, func_info) in functions {
        for (handle, expr) in func.expressions.iter() {
            let naga::Expression::Binary { op, right, .. } = *expr else {
                continue;
            };
            let op = match op {
                naga::BinaryOperator::Divide => "/",
                naga::BinaryOperator::Modulo => "%",
                _ => continue,
            };
            let sign = match func_info[handle].ty.inner_with(&module.types).scalar_kind() {
                Some(naga::ScalarKind::Uint) => Sign::Unsigned,
                Some(naga::ScalarKind::Sint) => Sign::Signed,
                _ => continue,
            };
            let span = func.expressions.get_span(right).to_range();
            let divisor = span
                .clone()
                .map(|r| wgsl[r].split_whitespace().collect::<Vec<_>>().join(" "))
                .unwrap_or_else(|| "<no span>".to_string());
            let line = func
                .expressions
                .get_span(handle)
                .to_range()
                .map(|r| wgsl[..r.start].matches('\n').count() + 1)
                .unwrap_or(0);
            sites.push(Site {
                source: source.to_string(),
                op,
                divisor,
                sign,
                constant: const_value(&module, &func.expressions, right),
                line,
            });
        }
    }
    Ok(sites)
}

/// Audit classes, as defined in the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// Nonzero constant.
    A,
    /// Runtime value the host guarantees nonzero.
    B,
    /// Runtime value the shader itself proves nonzero (dominating check or
    /// arithmetic identity).
    S,
}

#[derive(Debug)]
struct Row {
    count: usize,
    class: Class,
}

/// Key of a divisor-table row.
type Key = (String, String, String);

struct Audit {
    rows: BTreeMap<Key, Row>,
    not_compiled: Vec<String>,
}

/// Cells of a markdown table row, with surrounding backticks stripped.
fn cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split(" | ")
        .map(|c| c.trim().trim_matches('`').replace("\\|", "|"))
        .collect()
}

/// Lines of the markdown section that starts with `heading`.
fn section<'a>(doc: &'a str, heading: &str) -> Vec<&'a str> {
    let mut lines = doc.lines().skip_while(|l| l.trim() != heading);
    assert!(
        lines.next().is_some(),
        "{AUDIT_DOC} has no `{heading}` section"
    );
    lines.take_while(|l| !l.starts_with("## ")).collect()
}

fn read_audit() -> Audit {
    let path = workspace_root().join(AUDIT_DOC);
    let doc = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut rows = BTreeMap::new();
    for line in section(&doc, "## Divisor table") {
        if !line.starts_with("| `crates/") {
            continue;
        }
        let c = cells(line);
        assert!(c.len() >= 6, "malformed divisor row: {line}");
        let class = match c[4].as_str() {
            "A" => Class::A,
            "B" => Class::B,
            "S" => Class::S,
            other => panic!("unknown class `{other}` in row: {line}"),
        };
        let count = c[3]
            .parse()
            .unwrap_or_else(|_| panic!("bad count in row: {line}"));
        let key = (c[0].clone(), c[1].clone(), c[2].clone());
        assert!(
            rows.insert(key, Row { count, class }).is_none(),
            "duplicate row: {line}"
        );
    }

    let not_compiled = section(&doc, "## Sources naga rejects")
        .into_iter()
        .filter(|l| l.starts_with("| `crates/"))
        .map(|l| cells(l)[0].clone())
        .collect();

    Audit { rows, not_compiled }
}

fn suggested_row(key: &Key, sites: &[&Site]) -> String {
    let class = if sites[0].constant.is_some() {
        "A"
    } else {
        "?"
    };
    let mut lines: Vec<usize> = sites.iter().map(|s| s.line).collect();
    lines.sort_unstable();
    let lines: Vec<String> = lines.iter().map(usize::to_string).collect();
    format!(
        "| `{}` | `{}` | `{}` | {} | {class} | (lines {}) |",
        key.0,
        key.1,
        key.2.replace('|', "\\|"),
        sites.len(),
        lines.join(", ")
    )
}

#[test]
fn every_integer_division_is_in_the_audit() {
    let audit = read_audit();
    let corpus = corpus();
    assert!(
        corpus.len() >= 110,
        "corpus shrank to {} sources - the scanner lost a directory",
        corpus.len()
    );

    let mut sites: BTreeMap<Key, Vec<Site>> = BTreeMap::new();
    let mut rejected = Vec::new();
    for (source, wgsl) in &corpus {
        match scan(source, wgsl) {
            Ok(found) => {
                for s in found {
                    let key = (s.source.clone(), s.op.to_string(), s.divisor.clone());
                    sites.entry(key).or_default().push(s);
                }
            }
            Err(diag) => rejected.push((source.clone(), diag)),
        }
    }

    let mut problems = Vec::new();

    // Sources naga rejects never reach a pipeline, so they cannot be scanned.
    // They are pinned so that fixing one forces it into the audit proper.
    let rejected_names: Vec<&String> = rejected.iter().map(|(s, _)| s).collect();
    for (source, diag) in &rejected {
        if !audit.not_compiled.contains(source) {
            problems.push(format!(
                "{source} does not parse/validate and is not listed under \
                 `Sources naga rejects`:\n{diag}"
            ));
        }
    }
    for source in &audit.not_compiled {
        if !rejected_names.contains(&source) {
            problems.push(format!(
                "{source} is listed under `Sources naga rejects` but now \
                 validates (or no longer exists): audit its divisions and \
                 move it into the divisor table"
            ));
        }
    }

    for (key, found) in &sites {
        let refs: Vec<&Site> = found.iter().collect();
        let Some(row) = audit.rows.get(key) else {
            problems.push(format!(
                "unaudited integer division, add a row:\n  {}",
                suggested_row(key, &refs)
            ));
            continue;
        };
        if row.count != found.len() {
            problems.push(format!(
                "count changed for {key:?}: audit says {}, shader has {}:\n  {}",
                row.count,
                found.len(),
                suggested_row(key, &refs)
            ));
        }
        for site in found {
            match (&site.constant, row.class) {
                (Some(values), Class::A) => {
                    for &v in values {
                        let bad = v == 0 || (site.sign == Sign::Signed && v == -1);
                        if bad {
                            problems.push(format!(
                                "{}:{}: constant divisor `{}` is {v}",
                                site.source, site.line, site.divisor
                            ));
                        }
                    }
                }
                (None, Class::A) => problems.push(format!(
                    "{}:{}: divisor `{}` is classed A but is not a constant",
                    site.source, site.line, site.divisor
                )),
                (Some(_), _) => problems.push(format!(
                    "{}:{}: divisor `{}` is a constant, class it A",
                    site.source, site.line, site.divisor
                )),
                (None, Class::B | Class::S) => {}
            }
        }
    }
    for key in audit.rows.keys() {
        if !sites.contains_key(key) {
            problems.push(format!("stale audit row, no such division: {key:?}"));
        }
    }

    assert!(
        problems.is_empty(),
        "{AUDIT_DOC} is out of date ({} problem(s)):\n\n{}",
        problems.len(),
        problems.join("\n\n")
    );
}

/// The scanner must see through compound assignment, vectors and named
/// constants, and must ignore float division - otherwise an empty audit
/// would pass vacuously.
#[test]
fn scanner_finds_integer_divisions_only() {
    let src = r#"
const FOUR: u32 = 4u;
struct P { n: u32, d: i32 }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    var a = g.x / FOUR;
    a %= p.n;
    let v = vec2<u32>(g.x, g.y) / vec2<u32>(2u, 3u);
    let s = i32(g.x) % p.d;
    let f = f32(g.x) / 3.0;
    out[g.x] = a + v.x + u32(s) + u32(f);
}
"#;
    let mut sites = scan("t", src).unwrap();
    sites.sort_by_key(|s| s.line);
    let got: Vec<(&str, &str, Option<Vec<i64>>)> = sites
        .iter()
        .map(|s| (s.op, s.divisor.as_str(), s.constant.clone()))
        .collect();
    assert_eq!(
        got,
        vec![
            ("/", "FOUR", Some(vec![4])),
            ("%", "p.n", None),
            ("/", "vec2<u32>(2u, 3u)", Some(vec![2, 3])),
            ("%", "p.d", None),
        ]
    );
    assert_eq!(sites[3].sign, Sign::Signed);
}

/// Inline shaders are keyed by the Rust binding that holds them, which is
/// what the audit table names.
#[test]
fn inline_shaders_are_named_by_their_binding() {
    let rs = "fn a() {}\n    let voxel_shader = r#\"@compute\"#;\npub fn k() -> &'static str {\n    r#\"x\"#\n}";
    let lits = raw_string_literals(rs);
    assert_eq!(lits[0].0, "voxel_shader");
    assert_eq!(lits[1].0, "k");
}

const SIGNED_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<i32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= arrayLength(&a)) { return; }
    out[i] = vec2<i32>(a[i] % b[i], a[i] / b[i]);
}
"#;

/// The border-folding shaders (`convolve_2d`, `gaussian_blur_separable`,
/// `remap`, `undistort`) apply `%` to negative coordinates and rely on WGSL's
/// truncated semantics (the result takes the dividend's sign). Commit 9c14f40
/// replaced such a fold in `pyramid_down.wgsl` after it misbehaved on this
/// backend, so pin the semantics on the real device, with naga's
/// integer-division guard on (the default) and off. Skips without an adapter.
#[test]
fn signed_division_of_negatives_is_truncated() {
    use wgpu::util::DeviceExt;

    let Ok(ctx) = pollster::block_on(cv_hal::gpu::GpuContext::init_global()) else {
        println!("skipping: no GPU adapter");
        return;
    };
    let device = &ctx.device;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in -20i32..=20 {
        for d in [1, 2, 3, 5, 6, 7, 8, 14, -3] {
            a.push(x);
            b.push(d);
        }
    }
    let want: Vec<i32> = a
        .iter()
        .zip(&b)
        .flat_map(|(&x, &d)| [x.wrapping_rem(d), x.wrapping_div(d)])
        .collect();

    for int_div_checks in [true, false] {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some("signed division probe"),
            source: wgpu::ShaderSource::Wgsl(SIGNED_PROBE.into()),
        };
        let module = if int_div_checks {
            device.create_shader_module(desc)
        } else {
            let checks = wgpu::ShaderRuntimeChecks {
                int_div_checks: false,
                ..wgpu::ShaderRuntimeChecks::checked()
            };
            // SAFETY: every divisor in `b` is nonzero, and no dividend is
            // INT_MIN.
            unsafe { device.create_shader_module_trusted(desc, checks) }
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let input = |data: &[i32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let (a_buf, b_buf) = (input(&a), input(&b));
        let size = (want.len() * 4) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((a.len() as u32).div_ceil(64), 1, 1);
        }
        ctx.submit(enc);
        let got: Vec<i32> = pollster::block_on(cv_hal::gpu_kernels::buffer_utils::read_buffer(
            device.clone(),
            &ctx.queue,
            &out,
            0,
            size as usize,
        ))
        .unwrap();
        for (i, (g, w)) in got.chunks(2).zip(want.chunks(2)).enumerate() {
            assert_eq!(
                g,
                w,
                "int_div_checks = {int_div_checks}: ({x} % {d}, {x} / {d}) gave {g:?}, want \
                 {w:?} (truncated)",
                x = a[i],
                d = b[i],
            );
        }
    }
}
