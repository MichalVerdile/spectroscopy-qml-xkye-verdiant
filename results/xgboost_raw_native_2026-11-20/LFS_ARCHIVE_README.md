# Complete RunPod training artifacts

The source experiment directory, generated splits and generated data caches are archived in this branch. Models and binary outputs use Git LFS. Run git lfs pull after cloning.

Two cache files exceed the per-file limit for GitHub Free/Pro and are stored as numbered LFS parts. Reconstruct them from the repository root using the following commands. Verify their SHA-256 digests against lfs_artifact_manifest.json.

cat results/xgboost_raw_native_2026-11-20/generated_cache/shared_hnmr_len10000_40d66613724d2476.npz.parts/part-* > results/xgboost_raw_native_2026-11-20/generated_cache/shared_hnmr_len10000_40d66613724d2476.npz
Expected SHA-256 32b584ced8528dc35b48e55814e84a39d1a4476d481e1e532f624d5b271f4de6
cat results/xgboost_raw_native_2026-11-20/generated_cache/shared_ir_len1800_898a23afdff1f8e1.npz.parts/part-* > results/xgboost_raw_native_2026-11-20/generated_cache/shared_ir_len1800_898a23afdff1f8e1.npz
Expected SHA-256 a06c438d3d80d9c6495ed647696c68f0c0447b3894180683ffc37064e4e2f285