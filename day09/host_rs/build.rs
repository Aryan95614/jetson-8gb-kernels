// build.rs - links host_rs against libfused.so (the shim built from
// shim.cu) and against cudart. No external crates: this is std-only,
// using env::var and println! cargo directives.
//
// FUSED_DIR must be set (run.sh sets it to the day09 package root,
// which is where `nvcc ... -o libfused.so shim.cu` places the shared
// library) so rustc's linker can find -lfused.

use std::env;

fn main() {
    let fused_dir = env::var("FUSED_DIR")
        .expect("FUSED_DIR must be set to the directory containing libfused.so");

    println!("cargo:rustc-link-search=native={}", fused_dir);
    println!("cargo:rustc-link-lib=dylib=fused");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rerun-if-env-changed=FUSED_DIR");
}
