# 可选 FP64 稀疏代数

本目录属于 [rinkfit 历史后端](../../README.md)，当前 native LM 见[实现状态](../../../../docs/REBUILD.md)。

本目录的 cuSPARSE 包装库由 src/rinkfit_start/gpu_sparse.py 的 GpuCsr 显式启用。只做不可变完整矩阵 A @ x；不求解、分解、优化或采用模型状态。导入模块不初始化 CUDA。

当前支持 Windows、float64 数据、int32 索引、规范化 CSR。拒绝 float32、隐式索引缩窄、重复项及非有限值。使用 CUSPARSE_SPMV_CSR_ALG2 非转置运算；需要转置时调用方显式生成 CSR。上三角存储只代表上三角，不自动补对称、不改单位、不减行或裁 W。矩阵上传一次，每次传入并传回完整向量；错误显式报告，无静默 CPU 替代。

构建使用本目录 build.py，必需参数为 --cuda-root、--vcvars、--out-dir，可用 --sdk-root 选择现有 SDK 根目录。输出目录必须位于 OneDrive 项目之外。当前验证的目录为 D:/Program Files/CUDA、D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat 和 D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/gpu_accel/linear/native。

脚本直接调用现有 cl/link，显式提供现有 SDK 头文件和库，不安装依赖、不调用 nvcc。指定安装内存在多个 MSVC/SDK 版本时显式报错。build.json 记录源码/DLL SHA256、命令、编译选项和 CUDA 版本。

Python 使用：

    from rinkfit_start.gpu_sparse import GpuCsr
    with GpuCsr(A_csr, library_path=absolute_dll_path, cuda_bin=absolute_cuda_bin) as gpu:
        y = gpu.matvec(x_float64)

A_csr 需由调用方核验为完整原 float64/CSR32 算子。也可用 RINKFIT_GPU_SPARSE_DLL 显式指定 DLL；没有默认启用或拟合器接线。

2026-10-18 完整冻结系统验证在 D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/gpu_accel/linear/。benchmark_api.py / benchmark_api.json 记录原 A/P/KW/raw_W_J 哈希、全部 182016 source norms、完整 214112 行 W、原 W 单位残差/成本/梯度和传输计时。旧 MaxIterations 向量只作为固定算术输入，仍未采纳。数值一致性不代表收敛或物理精度。

驻矩阵重复乘法的收益不能外推为 Clarabel 的 281 秒求解加速；首次转换、上传及初始化须单独计入。未接入稀疏直接求解或 Gram 装配。现有 cuSOLVER 的 Cholesky/QR/host-LU 不能作为完整锥系统原 KKT 路径的等价替换；当前已核验 CUDA 安装没有 cuDSS。完整锥结构、primary/secondary 约束、单位和收敛判据保持原有实现；本接口不将其改成 LM/SPD 问题。
