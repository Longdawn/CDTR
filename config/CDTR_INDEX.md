# 配置索引
- 当前匹配检测预训练：MTHV2_stage1_query_budget.py；以冻结运行目录的配置与显式 options 为准。
- MTHv2 识别：SAQT_MTHV2.py / MTHV2_dtlr.py。
- HDRC 跨域：SAQT_HDRC.py；低资源子集为 SAQT_HDRC_low_resource.py。
- 历史源预训练：MTH1000_MTH1200_stage1.py。
- 含 probe、qclc、soft_order、geometry 等的文件为实验分支，不能默认并入当前方法。
- Synthetic、SynthVerticalAncient 等文件保留为历史合成数据实验。
旧配置不统一改名；实际训练协议以各输出目录 config_args_all.json 及运行命令为准。
