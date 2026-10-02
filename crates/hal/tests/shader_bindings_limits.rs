//! Static check: no WGSL entry point may use more than four storage buffers.
//!
//! The device limits in `gpu::GpuContext::init` come from
//! `wgpu::Limits::downlevel_defaults()`, which sets
//! `max_storage_buffers_per_shader_stage: 4`. Any entry point that references
//! more than four storage buffers therefore cannot have a pipeline created:
//! `create_compute_pipeline` fails with
//! `Too many bindings of type StorageBuffers, limit is 4`.
//!
//! wgpu-core derives the bind group layout **per entry point**, walking only the
//! global variables that entry point actually reaches (directly or through a
//! called function). Declarations that no entry point touches cost nothing, so
//! what matters is the *reachable* count per entry point, not the number of
//! `var<storage>` lines in the file.
//!
//! This test is purely static: it parses the WGSL text and walks the call graph
//! itself. It needs no adapter, and it is what would have caught every instance
//! of this bug class - `create_shader_module` alone performs no layout
//! derivation, so "the shader compiles" says nothing about binding counts.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// `wgpu::Limits::downlevel_defaults().max_storage_buffers_per_shader_stage`.
const MAX_STORAGE_PER_STAGE: usize = 4;

// ---------------------------------------------------------------------------
// WGSL parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressSpace {
    Storage,
    Uniform,
}

#[derive(Debug, Clone)]
struct Binding {
    group: u32,
    binding: u32,
    name: String,
    space: AddressSpace,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    /// Global variable names mentioned anywhere in the body.
    globals_used: BTreeSet<String>,
    /// Names of functions called from the body.
    calls: BTreeSet<String>,
    is_entry_point: bool,
    line: usize,
}

#[derive(Debug)]
struct Module {
    bindings: Vec<Binding>,
    functions: BTreeMap<String, Function>,
}

impl Module {
    /// Storage bindings reachable from `entry`, as `(group, binding)`.
    fn reachable_storage(&self, entry: &str) -> Vec<(u32, u32)> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut used: BTreeSet<String> = BTreeSet::new();
        let mut queue = vec![entry.to_string()];

        while let Some(name) = queue.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let Some(func) = self.functions.get(&name) else {
                continue;
            };
            used.extend(func.globals_used.iter().cloned());
            queue.extend(func.calls.iter().cloned());
        }

        self.bindings
            .iter()
            .filter(|b| b.space == AddressSpace::Storage && used.contains(&b.name))
            .map(|b| (b.group, b.binding))
            .collect()
    }

    fn entry_points(&self) -> Vec<&Function> {
        let mut eps: Vec<&Function> = self
            .functions
            .values()
            .filter(|f| f.is_entry_point)
            .collect();
        eps.sort_by(|a, b| a.name.cmp(&b.name));
        eps
    }
}

/// Remove `//` and `/* */` comments, preserving line structure and byte length
/// inside lines so column arithmetic stays meaningful.
fn strip_comments(src: &str) -> String {
    let bytes: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    let mut in_block = false;

    while i < bytes.len() {
        if in_block {
            if bytes[i] == '*' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
                in_block = false;
                out.push(' ');
                i += 2;
                continue;
            }
            if bytes[i] == '\n' {
                out.push('\n');
            }
            i += 1;
            continue;
        }
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
            while i < bytes.len() && bytes[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
            in_block = true;
            out.push(' ');
            out.push(' ');
            i += 2;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// All module-scope `var<storage, ...>` / `var<uniform>` declarations, with the
/// `@group`/`@binding` attributes that precede them.
fn parse_bindings(src: &str) -> Vec<Binding> {
    let cleaned = strip_comments(src);
    let bytes: Vec<char> = cleaned.chars().collect();
    let mut bindings = Vec::new();

    // Track brace depth so that only module-scope declarations are considered,
    // and remember the most recent @group/@binding pair seen at depth 0.
    let mut depth: i32 = 0;
    let mut pending: Option<(u32, u32)> = None;
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i];

        if c == '{' {
            depth += 1;
            i += 1;
            continue;
        }
        if c == '}' {
            depth -= 1;
            if depth < 0 {
                depth = 0;
            }
            i += 1;
            continue;
        }

        if c == '@' && depth == 0 {
            let rest: String = bytes[i..].iter().collect();
            let (attr_len, group, binding) = parse_attribute(&rest);
            if attr_len > 0 {
                if let (Some(g), Some(b)) = (group, binding) {
                    pending = Some((g, b));
                } else {
                    // Some other attribute (`@align`, `@id`, ...): the @group /
                    // @binding pair still applies to the declaration that
                    // follows, so leave `pending` alone.
                }
                i += attr_len;
                continue;
            }
            i += 1;
            continue;
        }

        if depth == 0 && (c == 'v' || c == 'V') && starts_with_word(&bytes, i, "var") {
            let rest: String = bytes[i..].iter().collect();
            if let Some((space, var_name, _)) = parse_var_decl(&rest) {
                if let Some((group, bind)) = pending {
                    bindings.push(Binding {
                        group,
                        binding: bind,
                        name: var_name,
                        space,
                    });
                    pending = None;
                }
            }
            i += 1;
            continue;
        }

        // Reset a pending attribute if some other declaration intervenes.
        if depth == 0 && pending.is_some() && c == ';' {
            pending = None;
        }
        i += 1;
    }

    bindings
}

fn starts_with_word(chars: &[char], i: usize, word: &str) -> bool {
    let w: Vec<char> = word.chars().collect();
    if i + w.len() > chars.len() {
        return false;
    }
    if chars[i..i + w.len()] != w[..] {
        return false;
    }
    if i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
        return false;
    }
    true
}

/// `fn name(...)` declarations plus what each body mentions.
fn parse_functions(src: &str) -> BTreeMap<String, Function> {
    let cleaned = strip_comments(src);
    let chars: Vec<char> = cleaned.chars().collect();
    let mut functions: BTreeMap<String, Function> = BTreeMap::new();

    // Line number of each char index, for diagnostics.
    let mut line_of: Vec<usize> = Vec::with_capacity(chars.len());
    let mut line = 1usize;
    for &c in &chars {
        line_of.push(line);
        if c == '\n' {
            line += 1;
        }
    }

    let mut i = 0usize;
    let mut depth: i32 = 0;
    let mut pending_compute = false;
    // Any `@compute` seen at depth 0 since the last `fn`.
    let mut saw_compute = false;

    while i < chars.len() {
        let c = chars[i];

        if c == '{' {
            depth += 1;
            i += 1;
            continue;
        }
        if c == '}' {
            depth -= 1;
            if depth < 0 {
                depth = 0;
            }
            i += 1;
            continue;
        }

        if depth == 0 && c == '@' {
            if starts_with_word(&chars, i, "@compute") {
                saw_compute = true;
                pending_compute = true;
            }
            i += 1;
            continue;
        }

        if depth == 0 && c == 'f' && starts_with_word(&chars, i, "fn") {
            let rest: String = chars[i..].iter().collect();
            let name = parse_fn_name(&rest);
            // Find the body: from here to the matching closing brace.
            let start = i;
            let mut j = i;
            let mut body_depth = 0i32;
            let mut body_end = chars.len();
            let mut opened = false;
            while j < chars.len() {
                if chars[j] == '{' {
                    body_depth += 1;
                    opened = true;
                } else if chars[j] == '}' {
                    body_depth -= 1;
                    if opened && body_depth == 0 {
                        body_end = j;
                        break;
                    }
                }
                j += 1;
            }
            let body: String = chars[start..body_end].iter().collect();
            let (globals_used, calls) = scan_body(&body);
            let func = Function {
                name: name.clone(),
                globals_used,
                calls,
                is_entry_point: pending_compute,
                line: line_of.get(start).copied().unwrap_or(1),
            };
            functions.insert(name, func);
            pending_compute = false;
            saw_compute = false;
            i = body_end.max(i + 1);
            continue;
        }

        if depth == 0 && (c == ';' || c == '}') {
            pending_compute = false;
            saw_compute = false;
        }
        let _ = saw_compute;
        i += 1;
    }

    functions
}

/// Parse `@group(g) @binding(b)` (attributes in either order) starting at `rest`.
/// Returns `(bytes consumed, group, binding)`.
fn parse_attribute(rest: &str) -> (usize, Option<u32>, Option<u32>) {
    let mut group = None;
    let mut binding = None;
    let mut pos = 0usize;
    let bytes: Vec<char> = rest.chars().collect();

    while pos < bytes.len() && bytes[pos] == '@' {
        let tail: String = bytes[pos..].iter().collect();
        let token = tail.split_whitespace().next().unwrap_or("").to_string();
        let name = token.trim_start_matches('@');
        let name = match name.find('(') {
            Some(p) => &name[..p],
            None => name,
        };
        let value = parse_u32_after(&tail, name);
        match name {
            "group" => group = value,
            "binding" => binding = value,
            _ => break,
        }
        pos += token.chars().count();
        // Skip whitespace between attributes.
        while pos < bytes.len() && (bytes[pos] == ' ' || bytes[pos] == '\t') {
            pos += 1;
        }
    }

    if group.is_none() && binding.is_none() {
        (0, None, None)
    } else {
        (pos, group, binding)
    }
}

fn parse_u32_after(text: &str, keyword: &str) -> Option<u32> {
    let idx = text.find(keyword)?;
    let tail = &text[idx + keyword.len()..];
    let tail = tail.trim_start();
    let open = tail.find('(')?;
    let inner = &tail[open + 1..];
    let end = inner.find(')')?;
    inner[..end].trim().parse::<u32>().ok()
}

/// Parse `var<storage, read_write> name: type;` or `var<uniform> name: type;`.
/// Returns `(space, name, bytes consumed)`.
fn parse_var_decl(rest: &str) -> Option<(AddressSpace, String, usize)> {
    if !rest.starts_with("var") {
        return None;
    }
    let lt = rest.find('<')?;
    let gt = rest.find('>')?;
    let inner = &rest[lt + 1..gt];
    let space = if inner.starts_with("storage") {
        AddressSpace::Storage
    } else if inner.starts_with("uniform") {
        AddressSpace::Uniform
    } else {
        // var<private>, var<workgroup>, ... - not a resource binding.
        return None;
    };
    let after = &rest[gt + 1..];
    let colon = after.find(':')?;
    let name: String = after[..colon]
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    Some((space, name, gt + 1))
}

fn parse_fn_name(rest: &str) -> String {
    let after = &rest[2..];
    let end = after.find('(').unwrap_or(after.len());
    after[..end]
        .trim()
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '_')
        .collect()
}

/// Collect the identifiers a function body could reach: every call target and
/// every plain identifier (identifiers are filtered against the known globals
/// and functions by the caller, so over-collecting here is safe).
fn scan_body(body: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let chars: Vec<char> = body.chars().collect();
    let mut idents: BTreeSet<String> = BTreeSet::new();
    let mut calls: BTreeSet<String> = BTreeSet::new();

    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '(' {
                calls.insert(word);
            } else {
                idents.insert(word);
            }
            continue;
        }
        i += 1;
    }

    // A called function's own name also appears as an identifier at the call
    // site; merge both so the closure in `reachable_storage` sees everything.
    idents.extend(calls.iter().cloned());
    (idents, calls)
}

fn parse_module(src: &str) -> Module {
    let bindings = parse_bindings(src);
    let mut functions = parse_functions(src);

    // Intersect each body's identifiers with the module's global bindings and
    // function names, so the closure only walks real edges.
    let binding_names: BTreeSet<String> = bindings.iter().map(|b| b.name.clone()).collect();
    let fn_names: BTreeSet<String> = functions.keys().cloned().collect();
    for func in functions.values_mut() {
        func.globals_used.retain(|n| binding_names.contains(n));
        func.globals_used
            .extend(func.calls.iter().filter(|n| fn_names.contains(*n)).cloned());
        func.calls.retain(|n| fn_names.contains(n));
    }

    Module {
        bindings,
        functions,
    }
}

// ---------------------------------------------------------------------------
// Shader discovery
// ---------------------------------------------------------------------------

fn hal_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn all_wgsl_files() -> Vec<PathBuf> {
    let root = hal_root();
    let mut out = Vec::new();
    for dir in ["shaders", "src/gpu_kernels"] {
        collect_wgsl(&root.join(dir), &mut out);
    }
    out.sort();
    out
}

fn collect_wgsl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
            out.push(path);
        }
    }
}

/// Inline WGSL lives in Rust raw string literals (`r#"..."#`) inside the kernel
/// modules; those shaders are compiled by exactly the same path as the `.wgsl`
/// files, so they get the same treatment.
fn inline_wgsl_sources() -> Vec<(String, String)> {
    let dir = hal_root().join("src/gpu_kernels");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        for (idx, chunk) in raw_string_literals(&text).into_iter().enumerate() {
            if chunk.contains("@compute") {
                out.push((format!("{name}.rs:inline#{idx}"), chunk));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn raw_string_literals(text: &str) -> Vec<String> {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 'r' && i + 1 < bytes.len() && bytes[i + 1] == '#' {
            let mut hashes = 0usize;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < bytes.len() && bytes[j] == '"' {
                let body_start = j + 1;
                let terminator: String = format!("\"{}", "#".repeat(hashes));
                if let Some(rel_end) = text[body_start..].find(&terminator) {
                    let body: String = text[body_start..body_start + rel_end].to_string();
                    out.push(body);
                    i = body_start + rel_end + terminator.len();
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The core invariant: every entry point in every shader uses at most
/// `max_storage_buffers_per_shader_stage` storage buffers.
#[test]
fn every_entry_point_fits_the_storage_buffer_limit() {
    let mut sources: Vec<(String, String)> = Vec::new();
    for path in all_wgsl_files() {
        let rel = path
            .strip_prefix(hal_root())
            .unwrap_or(&path)
            .display()
            .to_string();
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        sources.push((rel, text));
    }
    sources.extend(inline_wgsl_sources());

    assert!(
        sources.len() >= 100,
        "expected the whole shader corpus to be covered, found only {} sources",
        sources.len()
    );

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut violations: Vec<String> = Vec::new();
    let mut total_entry_points = 0usize;

    for (name, src) in &sources {
        let module = parse_module(src);
        for ep in module.entry_points() {
            total_entry_points += 1;
            let storage = module.reachable_storage(&ep.name);
            *counts.entry(name.clone()).or_insert(0) += 1;
            if storage.len() > MAX_STORAGE_PER_STAGE {
                let layout: Vec<String> = storage
                    .iter()
                    .map(|(g, b)| format!("@group({g}) @binding({b})"))
                    .collect();
                violations.push(format!(
                    "{name}:{}: entry point `{}` reaches {} storage buffers \
                     (limit {MAX_STORAGE_PER_STAGE}): {}",
                    ep.line,
                    ep.name,
                    storage.len(),
                    layout.join(", ")
                ));
            }
        }
    }

    assert!(
        total_entry_points > 0,
        "no entry points were parsed - the scanner is broken, not the shaders"
    );
    assert!(
        violations.is_empty(),
        "{} entry point(s) exceed the {MAX_STORAGE_PER_STAGE}-storage-buffer-per-stage \
         limit (limits come from wgpu::Limits::downlevel_defaults()):\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
}

/// Guards the scanner itself: the known-good LBVH split must parse with the
/// per-entry-point counts the layout derivation actually produces, proving the
/// walk is doing real work and not just counting declarations.
#[test]
fn scanner_counts_reachable_bindings_per_entry_point_not_per_file() {
    let root = hal_root();
    let build = std::fs::read_to_string(root.join("shaders/lbvh_build.wgsl")).unwrap();
    let aabb = std::fs::read_to_string(root.join("shaders/lbvh_aabb.wgsl")).unwrap();

    let m_build = parse_module(&build);
    assert!(
        m_build.bindings.len() > 2,
        "lbvh_build.wgsl should declare more storage bindings than the tree phase uses"
    );
    let init = m_build.reachable_storage("init_nodes");
    let tree = m_build.reachable_storage("build_radix_tree");
    // `build_radix_tree` reads morton_codes via `delta` and writes nodes;
    // `init_nodes` only writes nodes, and that is exactly why the host builds
    // it a two-entry bind group rather than the three the layout would derive
    // for the other pipeline.
    assert_eq!(
        init,
        vec![(0, 1)],
        "lbvh_build.wgsl `init_nodes` should reach only `nodes`"
    );
    assert_eq!(
        tree,
        vec![(0, 0), (0, 1)],
        "lbvh_build.wgsl `build_radix_tree` should reach morton_codes + nodes"
    );

    let m_aabb = parse_module(&aabb);
    let compute_aabbs = m_aabb.reachable_storage("compute_aabbs");
    assert_eq!(
        compute_aabbs,
        vec![(0, 0), (0, 1), (0, 2), (0, 3)],
        "lbvh_aabb.wgsl `compute_aabbs` should reach all four storage buffers"
    );
}

/// Helper functions must be followed: a binding used only inside a `fn` that
/// the entry point calls still counts. `matrix_multiply.wgsl`-style helper
/// indirection is exactly the case a naive line-based scan gets wrong.
#[test]
fn scanner_follows_helper_functions() {
    let src = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<f32>;

fn load(idx: u32) -> f32 {
    return a[idx] + b[idx] + p.x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    c[gid.x] = load(gid.x);
}
"#;
    let m = parse_module(src);
    let eps = m.entry_points();
    assert_eq!(eps.len(), 1, "expected exactly one entry point");
    assert_eq!(m.reachable_storage("main").len(), 3);
}

/// An entry point that ignores a declared binding must not be charged for it -
/// this is the property the disproven "per-module layout" model got backwards,
/// and several files rely on it.
#[test]
fn scanner_ignores_unreferenced_declarations() {
    let src = r#"
@group(0) @binding(0) var<storage, read> used: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<storage, read> unused: array<f32>;
@group(0) @binding(3) var<storage, read> also_unused: array<f32>;
@group(0) @binding(4) var<storage, read> never_used: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out_buf[gid.x] = used[gid.x];
}

@compute @workgroup_size(64)
fn other() {}
"#;
    let m = parse_module(src);
    assert_eq!(m.reachable_storage("main"), vec![(0, 0), (0, 1)]);
    assert!(m.reachable_storage("other").is_empty());
}

/// Uniform bindings are limited separately (`max_uniform_buffers_per_shader_stage`),
/// so they are reported but not asserted against the storage limit.
#[test]
fn scanner_reports_uniform_bindings_separately() {
    let src = r#"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> u1: vec4<f32>;
@group(0) @binding(2) var<uniform> u2: vec4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = data[gid.x] * u1.x * u2.x;
}
"#;
    let m = parse_module(src);
    assert_eq!(m.reachable_storage("main").len(), 1);
    let uniforms = m
        .bindings
        .iter()
        .filter(|b| b.space == AddressSpace::Uniform)
        .count();
    assert_eq!(uniforms, 2);
}

/// A storage buffer must never be bound at a `@group(1)`-style binding the host
/// forgets to supply; this is a cheap sanity check that the two host bind groups
/// the raycasting kernel builds are both described by the shader it includes.
#[test]
fn raycasting_shader_and_host_agree_on_bindings() {
    let root = hal_root();
    let shader =
        std::fs::read_to_string(root.join("src/gpu_kernels/ray_mesh_intersection.wgsl")).unwrap();
    let m = parse_module(&shader);

    let eps = m.entry_points();
    assert_eq!(eps.len(), 1, "expected exactly one entry point");

    let storage = m.reachable_storage("main");
    assert!(
        storage.len() <= MAX_STORAGE_PER_STAGE,
        "ray_mesh_intersection `main` reaches {} storage buffers: {storage:?}",
        storage.len()
    );

    // The host must supply every binding the shader declares, in the group it
    // declares it in. Missing entries are the *next* error after a count
    // violation, and they are just as fatal at bind-group creation.
    let host = std::fs::read_to_string(root.join("src/gpu_kernels/raycasting.rs")).unwrap();
    let mut derived_by_group: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for b in &m.bindings {
        derived_by_group
            .entry(b.group)
            .or_default()
            .insert(b.binding);
    }
    assert!(
        !derived_by_group.is_empty(),
        "no bindings parsed out of ray_mesh_intersection.wgsl - the scanner is broken"
    );

    for (group, bindings_in_group) in &derived_by_group {
        let group_block = bind_group_entries(&host, *group)
            .unwrap_or_else(|| panic!("raycasting.rs builds no bind group against layout {group}"));
        for binding in bindings_in_group {
            assert!(
                group_block.contains(&format!("binding: {binding},")),
                "raycasting.rs bind group {group} does not supply binding {binding}"
            );
        }
    }
}

/// Return the `entries: &[...]` body of every `create_bind_group` call in `src`
/// that is built against `get_bind_group_layout(group)`, concatenated.
fn bind_group_entries(src: &str, group: u32) -> Option<String> {
    let mut out = String::new();
    let mut found = false;
    let mut cursor = 0usize;
    while let Some(rel) = src[cursor..].find(&format!("get_bind_group_layout({group})")) {
        found = true;
        let after = cursor + rel;
        let rest = &src[after..];
        let Some(start) = rest.find("entries:") else {
            break;
        };
        let tail = &rest[start..];
        let Some(end) = tail.find("],") else {
            break;
        };
        out.push_str(&tail[..end]);
        out.push('\n');
        cursor = after + start + end;
    }
    if found {
        Some(out)
    } else {
        None
    }
}
/// Coverage guard: the corpus this file checks must be the whole shader set,
/// not just `shaders/*`. The original `every_shader_compiles`-style coverage
/// looked only at `shaders/`, and every file that actually violated the limit
/// lives in `src/gpu_kernels/`.
#[test]
fn the_whole_shader_corpus_is_covered() {
    let files = all_wgsl_files();
    assert_eq!(
        files.len(),
        104,
        "expected 104 .wgsl files across shaders/ and src/gpu_kernels/"
    );
    let kernels = files
        .iter()
        .filter(|p| p.to_string_lossy().contains("gpu_kernels"))
        .count();
    let shaders = files.len() - kernels;
    assert!(
        shaders > 0 && kernels > 0,
        "both directories must be covered"
    );

    // Inline `r#"..."#` shaders in the kernel .rs files are compiled by the
    // same path and were an entire class of violation the shaders/ sweep missed.
    let inline = inline_wgsl_sources();
    assert!(
        inline.len() >= 5,
        "expected the inline WGSL in src/gpu_kernels/*.rs to be found, got {}",
        inline.len()
    );

    // Sanity: every parsed module must yield at least one entry point, or the
    // limit assertion is vacuously passing on that file.
    for (name, src) in inline_wgsl_sources().into_iter().chain(
        files
            .iter()
            .map(|p| {
                (
                    p.file_stem().unwrap().to_string_lossy().to_string(),
                    std::fs::read_to_string(p).unwrap(),
                )
            })
            .collect::<Vec<_>>(),
    ) {
        let m = parse_module(&src);
        if m.bindings.is_empty() {
            continue; // A shader with no bindings at all.
        }
        assert!(
            !m.entry_points().is_empty(),
            "{name} declares bindings but no @compute entry point was parsed"
        );
    }
}
