import os
import re

deps_to_workspace = {
    "rayon", "nalgebra", "image", "wgpu", "bytemuck", "wide", "futures", "thiserror", "criterion",
    "core_affinity", "ndarray", "tokio", "ffmpeg-next", "rand", "serde", "serde_json", "pollster",
    "rstar", "itertools", "anyhow", "log", "tracing", "libc",
    "cv-core", "cv-hal", "cv-imgproc", "cv-features", "cv-stereo", "cv-video", "cv-3d", "cv-calib3d",
    "cv-sfm", "cv-slam", "cv-runtime", "cv-optimize", "cv-scientific", "cv-videoio",
    "cv-photo", "cv-io", "cv-objdetect", "cv-dnn", "cv-rendering", "cv-registration", "cv-plot", "cv-viewer"
}

def update_cargo_toml(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    changed = False
    
    # Simple regex to match dependencies like:
    # dep = "version"
    # dep = { version = "...", features = [...] }
    # dep = { path = "...", ... }
    
    for line in lines:
        match = re.match(r'^(\s*)([a-zA-Z0-9_-]+)(\s*=\s*)(.*)$', line)
        if match:
            indent, dep_name, eq, value = match.groups()
            if dep_name in deps_to_workspace:
                value = value.strip()
                # If it already has workspace = true, don't change it
                if "workspace = true" in value:
                    new_lines.append(line)
                    continue
                
                new_value = ""
                if value.startswith('"'):
                    # Case: dep = "version"
                    new_value = "{ workspace = true }"
                elif value.startswith('{'):
                    # Case: dep = { ... }
                    # Remove version or path, and add workspace = true
                    # Use regex to replace version="..." or path="..."
                    v = re.sub(r'version\s*=\s*"[^"]*",?\s*', '', value)
                    v = re.sub(r'path\s*=\s*"[^"]*",?\s*', '', v)
                    
                    # Ensure workspace = true is inside
                    if "workspace" not in v:
                        if v.strip() == "{}":
                            v = "{ workspace = true }"
                        else:
                            # Insert workspace = true at the beginning
                            v = v.replace('{', '{ workspace = true, ', 1)
                    
                    # Clean up trailing commas and double spaces
                    v = v.replace(', }', ' }').replace(',,', ',').replace('  ', ' ')
                    new_value = v
                else:
                    # Skip unknown formats
                    new_lines.append(line)
                    continue
                
                new_line = f"{indent}{dep_name}{eq}{new_value}\n"
                if new_line != line:
                    new_lines.append(new_line)
                    changed = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if changed:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")

sub_crates = [
    "core", "hal", "imgproc", "features", "stereo", "video", "videoio", "calib3d",
    "photo", "sfm", "slam", "runtime", "optimize", "scientific", "point-cloud",
    "python", "viewer", "io", "objdetect", "dnn", "3d", "rendering", "registration", "plot"
]

root_dir = "/home/prathana/RUST/rust-cv-native"
for crate in sub_crates:
    cargo_path = os.path.join(root_dir, crate, "Cargo.toml")
    if os.path.exists(cargo_path):
        update_cargo_toml(cargo_path)
    else:
        print(f"File not found: {cargo_path}")

# Also check benches and examples as they were mentioned in the workspace
update_cargo_toml(os.path.join(root_dir, "benches", "Cargo.toml"))
update_cargo_toml(os.path.join(root_dir, "examples", "Cargo.toml"))
