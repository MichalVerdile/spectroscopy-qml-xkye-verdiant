# Complete RunPod training artifacts

All 2,290 source files from the experiment directory, generated splits and generated data caches are archived in this branch. Models and binary outputs use Git LFS. Run the following after cloning.

```
git lfs pull
```

The two largest cache files are stored as numbered LFS parts. Reconstruct them from the repository root with the commands below. Compare the SHA-256 values with lfs_artifact_manifest.json.

```
cat results/xgboost_raw_native_2026-11-20/generated_cache/shared_hnmr_len10000_40d66613724d2476.npz.parts/part-* > results/xgboost_raw_native_2026-11-20/generated_cache/shared_hnmr_len10000_40d66613724d2476.npz
sha256sum results/xgboost_raw_native_2026-11-20/generated_cache/shared_hnmr_len10000_40d66613724d2476.npz
```

Expected SHA-256 is 32b584ced8528dc35b48e55814e84a39d1a4476d481e1e532f624d5b271f4de6.

```
cat results/xgboost_raw_native_2026-11-20/generated_cache/shared_ir_len1800_898a23afdff1f8e1.npz.parts/part-* > results/xgboost_raw_native_2026-11-20/generated_cache/shared_ir_len1800_898a23afdff1f8e1.npz
sha256sum results/xgboost_raw_native_2026-11-20/generated_cache/shared_ir_len1800_898a23afdff1f8e1.npz
```

Expected SHA-256 is a06c438d3d80d9c6495ed647696c68f0c0447b3894180683ffc37064e4e2f285.

Remote availability of all LFS objects and two complete model downloads were verified. See lfs_remote_verification.json.
