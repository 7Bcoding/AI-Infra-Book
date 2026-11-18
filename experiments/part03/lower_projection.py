"""Lower one 2x2 FP32 layer to NVPTX and assemble it for a real target.

Default mode checks the scalar contract using only Python. To regenerate the
compiler files, install llvmlite 0.46.0, put CUDA 12.9 ptxas on PATH, and run
python lower_projection.py --compile DIRECTORY. Add nvdisasm to PATH to emit SASS.
No device or driver is used.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

IR = """target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @llvm.fma.f32(float, float, float)
declare float @llvm.maxnum.f32(float, float)

define ptx_kernel void @projection(ptr addrspace(1) %x, ptr addrspace(1) %w,
                                   ptr addrspace(1) %b, ptr addrspace(1) %y) {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ok = icmp ult i32 %t, 4
  br i1 %ok, label %body, label %end
body:
  %col = and i32 %t, 1
  %rowbase = and i32 %t, 2
  %wi = add i32 %col, 2
  %xi = add i32 %rowbase, 1
  %px0 = getelementptr float, ptr addrspace(1) %x, i32 %rowbase
  %px1 = getelementptr float, ptr addrspace(1) %x, i32 %xi
  %pw0 = getelementptr float, ptr addrspace(1) %w, i32 %col
  %pw1 = getelementptr float, ptr addrspace(1) %w, i32 %wi
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %col
  %py = getelementptr float, ptr addrspace(1) %y, i32 %t
  %x0 = load float, ptr addrspace(1) %px0, align 4
  %x1 = load float, ptr addrspace(1) %px1, align 4
  %w0 = load float, ptr addrspace(1) %pw0, align 4
  %w1 = load float, ptr addrspace(1) %pw1, align 4
  %off = load float, ptr addrspace(1) %pb, align 4
  %first = fmul float %x0, %w0
  %sum = call float @llvm.fma.f32(float %x1, float %w1, float %first)
  %z = fadd float %sum, %off
  %out = call float @llvm.maxnum.f32(float %z, float 0.0)
  store float %out, ptr addrspace(1) %py, align 4
  br label %end
end:
  ret void
}
"""


def compile_files(folder):
    import llvmlite.binding as llvm
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    module = llvm.parse_assembly(IR)
    module.verify()
    machine = llvm.Target.from_triple(module.triple).create_target_machine(
        cpu="sm_80", features="+ptx70", opt=2)
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "layer.ll").write_text(IR, encoding="utf8")
    ptx = folder / "layer.ptx"
    ptx.write_text(machine.emit_assembly(module), encoding="utf8")
    assembler = shutil.which("ptxas")
    if not assembler:
        raise SystemExit("PTX saved. Add CUDA ptxas to PATH to produce the cubin.")
    binary = folder / "layer.cubin"
    subprocess.run([assembler, "-arch=sm_80", "-v", str(ptx), "-o", str(binary)], check=True)
    dumper = shutil.which("nvdisasm")
    if dumper:
        dump = subprocess.check_output([dumper, "-ndf", str(binary)])
        (folder / "layer.sass").write_bytes(dump)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", metavar="DIRECTORY", type=Path)
    options = parser.parse_args()
    x, w, bias = [1, 2, 3, 4], [2, -1, 1, 3], [-6, 1]
    output = []
    for thread in range(4):
        rowbase, col = thread & 2, thread & 1
        acc = x[rowbase] * w[col] + x[rowbase + 1] * w[col + 2]
        output.append(max(0, acc + bias[col]))
    assert output == [0, 6, 4, 10]
    print("Scalar check:", output)
    if options.compile:
        compile_files(options.compile)


if __name__ == "__main__":
    main()
