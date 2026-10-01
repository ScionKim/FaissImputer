# Available float64 selected-distance batching

Selected candidates are refined in bounded pair blocks using the existing guarded distance kernel. These archives compare that change and a subsequent same-code control.

## Sources and configuration

- **Before/after**: [a259000](https://github.com/ScionKim/FaissImputer/commit/a25900033ab590ae8c8982e0d996f03cbc26591c) → [c02b71d](https://github.com/ScionKim/FaissImputer/commit/c02b71d3290245d7a131a25e82f80fa028c086bf); [workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/36788192361); [raw archive](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-selected-distances-c02b71d.zip.zip).
  324 successful records; 108 matched baseline/candidate pairs. CPU: AMD EPYC 7763 64-Core Processor; Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.
- **Same-code control**: [c02b71d](https://github.com/ScionKim/FaissImputer/commit/c02b71d3290245d7a131a25e82f80fa028c086bf) → [6d63ae4](https://github.com/ScionKim/FaissImputer/commit/6d63ae4e2b5847c92e18b50e05a7a2de05633f6a); [workflow run](https://github.com/ScionKim/FaissImputer/actions/runs/36796444560); [raw archive](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-selected-distances-aa-6d63ae4.zip).
  324 successful records; 108 matched baseline/candidate pairs. CPU: AMD EPYC 7763 64-Core Processor; Python 3.12.14; NumPy 2.5.3; scikit-learn 1.9.1; Faiss 1.15.1.

Each case uses 20,000 training rows, 20 features, k=5, uniform weights, random query missingness, and one native thread. Complete-policy training is fully observed; available-policy training has a target missing rate of 10%. Query missingness is 20%. Therefore policy rows are distinct workloads.

## Aggregation definitions

All tables are recalculated from `records[]`; stored JSON summaries are not used. Only the `fit_then_transform` API is measured here. There is no same-data `fit_transform` comparison. Runs, dtypes, training policies and query sizes are never pooled.

Times are median [min–max] in seconds across 9 records = 3 seeds × 3 fresh-worker repeats. `total_seconds` is fit plus the first transform. `transform_seconds` is the first transform alone. The second and third calls are the two ordered entries in `repeated_transform_seconds`. The additional-call summary first takes their median within each worker, then summarizes those nine worker medians. It is not a median over 18 independent timing trials.

Each ratio is baseline time / candidate time for a pair matched within the same run, training configuration, policy, dtype, query size, seed and repeat, with identical input fingerprints. Reported ratios are median [min–max] of the nine paired ratios; values above 1 favor the candidate. They are not ratios of displayed median times. Each timing phase has its own ratios.

Peak RSS is the full worker-lifetime peak in MiB. Memory deltas are candidate minus baseline for each matched pair, summarized as median [min–max]; no memory speedup is defined. RMSE and MAE are errors against synthetic ground truth at the scored missing cells. Quality and donor counts use the same nine records; repeated seeds are not nine independently generated datasets.

## Before/after: timings

### First transform

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.157001 [0.151052–0.160014] | 0.156716 [0.150687–0.159300] | 1.004479 [0.964624–1.033114] |
| complete | float64 | 300 | 0.213080 [0.210227–0.218747] | 0.218578 [0.210626–0.221107] | 0.978804 [0.966312–1.009402] |
| available | float32 | 300 | 0.103969 [0.101338–0.107815] | 0.103128 [0.100096–0.120406] | 1.005881 [0.859461–1.050615] |
| available | float64 | 300 | 0.122662 [0.121331–0.129408] | 0.106447 [0.103261–0.110883] | 1.168467 [1.111833–1.187929] |
| complete | float32 | 1000 | 0.484616 [0.460406–0.491170] | 0.482170 [0.469049–0.491565] | 0.999196 [0.954862–1.037071] |
| complete | float64 | 1000 | 0.674998 [0.644765–0.688037] | 0.677343 [0.640157–0.690001] | 1.006628 [0.945849–1.020064] |
| available | float32 | 1000 | 0.340637 [0.326841–0.354491] | 0.340411 [0.322177–0.352265] | 1.011446 [0.967163–1.019096] |
| available | float64 | 1000 | 0.397833 [0.392768–0.416806] | 0.340023 [0.335933–0.355083] | 1.172912 [1.151903–1.204392] |
| complete | float32 | 3000 | 1.199226 [1.181466–1.272077] | 1.199866 [1.179773–1.244642] | 1.002971 [0.960582–1.049037] |
| complete | float64 | 3000 | 1.642931 [1.617858–1.782201] | 1.639170 [1.608495–1.754653] | 1.002294 [0.966339–1.047102] |
| available | float32 | 3000 | 0.954552 [0.926732–1.024906] | 0.973788 [0.937243–1.044900] | 0.994037 [0.948146–1.018468] |
| available | float64 | 3000 | 1.166497 [1.125465–1.193361] | 1.000025 [0.963109–1.026404] | 1.164477 [1.145939–1.207535] |

### Fit + first transform

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.162090 [0.156003–0.164975] | 0.161568 [0.155507–0.164372] | 1.003670 [0.965554–1.032752] |
| complete | float64 | 300 | 0.220306 [0.216768–0.225005] | 0.225911 [0.216970–0.228115] | 0.981907 [0.964075–1.008624] |
| available | float32 | 300 | 0.116329 [0.113882–0.121340] | 0.114538 [0.112316–0.131653] | 1.013942 [0.874913–1.064566] |
| available | float64 | 300 | 0.135390 [0.133319–0.143405] | 0.118685 [0.115264–0.123401] | 1.145613 [1.104381–1.185752] |
| complete | float32 | 1000 | 0.489492 [0.465182–0.496132] | 0.487030 [0.473744–0.496537] | 0.999184 [0.955142–1.037598] |
| complete | float64 | 1000 | 0.681512 [0.651714–0.695187] | 0.684000 [0.647540–0.696821] | 1.005860 [0.945280–1.020837] |
| available | float32 | 1000 | 0.352645 [0.338425–0.367104] | 0.352422 [0.333567–0.364699] | 1.013617 [0.966950–1.021083] |
| available | float64 | 1000 | 0.412372 [0.406616–0.430026] | 0.352670 [0.349395–0.369142] | 1.164727 [1.150613–1.197162] |
| complete | float32 | 3000 | 1.203937 [1.186270–1.277375] | 1.204544 [1.184479–1.249413] | 1.003009 [0.960844–1.049014] |
| complete | float64 | 3000 | 1.649725 [1.625478–1.789757] | 1.645508 [1.614949–1.762283] | 1.002563 [0.966443–1.047399] |
| available | float32 | 3000 | 0.965965 [0.938932–1.036973] | 0.986063 [0.947873–1.057789] | 0.994689 [0.948657–1.019087] |
| available | float64 | 3000 | 1.179702 [1.138011–1.207951] | 1.013238 [0.975113–1.039729] | 1.162336 [1.145473–1.207558] |

### Additional-call worker median

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.149637 [0.143606–0.154642] | 0.149627 [0.141653–0.154343] | 1.001937 [0.951406–1.047035] |
| complete | float64 | 300 | 0.204587 [0.202405–0.213711] | 0.212664 [0.200893–0.219007] | 0.957618 [0.934159–1.007525] |
| available | float32 | 300 | 0.097891 [0.094674–0.099002] | 0.095959 [0.094385–0.098294] | 1.009436 [0.987268–1.030034] |
| available | float64 | 300 | 0.117578 [0.114466–0.118866] | 0.098405 [0.097313–0.103906] | 1.177508 [1.143977–1.207130] |
| complete | float32 | 1000 | 0.477662 [0.452335–0.482343] | 0.477896 [0.457960–0.503848] | 0.999668 [0.897762–1.045077] |
| complete | float64 | 1000 | 0.666039 [0.636947–0.680408] | 0.666279 [0.639613–0.684003] | 0.999641 [0.943010–1.021569] |
| available | float32 | 1000 | 0.328441 [0.320957–0.340231] | 0.332642 [0.317384–0.341848] | 0.994092 [0.960780–1.011257] |
| available | float64 | 1000 | 0.392298 [0.388608–0.402681] | 0.333055 [0.322694–0.341595] | 1.184888 [1.166838–1.209847] |
| complete | float32 | 3000 | 1.192683 [1.173560–1.262946] | 1.192627 [1.174666–1.238585] | 1.013181 [0.962939–1.041860] |
| complete | float64 | 3000 | 1.639680 [1.606179–1.738940] | 1.816311 [1.768103–1.920272] | 0.903923 [0.871637–0.937097] |
| available | float32 | 3000 | 0.961579 [0.934977–1.033635] | 0.961652 [0.940438–1.032926] | 0.998126 [0.965764–1.064273] |
| available | float64 | 3000 | 1.163879 [1.126857–1.180715] | 0.985862 [0.957371–1.016881] | 1.172086 [1.140170–1.207556] |

## Same-code control: timings

### First transform

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.155881 [0.153128–0.164818] | 0.156596 [0.153120–0.160953] | 0.997844 [0.961709–1.076198] |
| complete | float64 | 300 | 0.217910 [0.212241–0.223186] | 0.213920 [0.211989–0.223059] | 1.007079 [0.976914–1.051117] |
| available | float32 | 300 | 0.111302 [0.106405–0.115214] | 0.110295 [0.106974–0.113716] | 1.006128 [0.968175–1.017926] |
| available | float64 | 300 | 0.111986 [0.109104–0.113716] | 0.110537 [0.109253–0.113517] | 1.014611 [0.970046–1.025128] |
| complete | float32 | 1000 | 0.472694 [0.469521–0.491326] | 0.471853 [0.467560–0.483918] | 1.004072 [0.984196–1.030052] |
| complete | float64 | 1000 | 0.660570 [0.645484–0.705299] | 0.658540 [0.647863–0.694230] | 1.003782 [0.929784–1.040535] |
| available | float32 | 1000 | 0.347411 [0.335556–0.349202] | 0.344284 [0.339928–0.354988] | 0.996358 [0.982612–1.015219] |
| available | float64 | 1000 | 0.348548 [0.345879–0.356516] | 0.348051 [0.342852–0.356360] | 1.005516 [0.976717–1.013422] |
| complete | float32 | 3000 | 1.255848 [1.213833–1.273680] | 1.252856 [1.196444–1.270922] | 1.001871 [0.955522–1.042829] |
| complete | float64 | 3000 | 1.746078 [1.626365–1.774679] | 1.750756 [1.659946–1.804913] | 0.994613 [0.951731–1.011147] |
| available | float32 | 3000 | 1.036087 [1.018468–1.063928] | 1.041095 [1.017026–1.058860] | 0.996231 [0.974490–1.018790] |
| available | float64 | 3000 | 1.059400 [1.025212–1.087469] | 1.063074 [1.030246–1.129061] | 0.997038 [0.950760–1.015859] |

### Fit + first transform

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.161078 [0.158113–0.169787] | 0.161609 [0.158103–0.165950] | 0.997977 [0.963553–1.073899] |
| complete | float64 | 300 | 0.224576 [0.218586–0.231472] | 0.220903 [0.219254–0.230208] | 1.005937 [0.975534–1.053333] |
| available | float32 | 300 | 0.123593 [0.118424–0.127744] | 0.122164 [0.119286–0.126423] | 1.009965 [0.973430–1.061691] |
| available | float64 | 300 | 0.124797 [0.121351–0.127576] | 0.123382 [0.121591–0.125681] | 1.013862 [0.975359–1.034288] |
| complete | float32 | 1000 | 0.477598 [0.474548–0.496367] | 0.476947 [0.472527–0.489051] | 1.004278 [0.984339–1.029984] |
| complete | float64 | 1000 | 0.667791 [0.651256–0.712635] | 0.665017 [0.655192–0.701322] | 1.002788 [0.928612–1.040964] |
| available | float32 | 1000 | 0.359784 [0.347260–0.361504] | 0.356694 [0.351595–0.366856] | 0.996968 [0.983178–1.015517] |
| available | float64 | 1000 | 0.361819 [0.359844–0.370001] | 0.361054 [0.355882–0.370654] | 1.004729 [0.973854–1.016550] |
| complete | float32 | 3000 | 1.260785 [1.218995–1.278668] | 1.257675 [1.201450–1.275841] | 1.001790 [0.955604–1.042670] |
| complete | float64 | 3000 | 1.752883 [1.633761–1.782221] | 1.758211 [1.667032–1.812745] | 0.994818 [0.951558–1.011184] |
| available | float32 | 3000 | 1.048659 [1.030798–1.076798] | 1.052177 [1.028127–1.071429] | 0.997275 [0.975161–1.019027] |
| available | float64 | 3000 | 1.072487 [1.038577–1.101914] | 1.075855 [1.043239–1.143538] | 0.997814 [0.951456–1.016387] |

### Additional-call worker median

| Policy | dtype | Queries | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- | --- |
| complete | float32 | 300 | 0.147863 [0.144214–0.152409] | 0.147349 [0.143836–0.151582] | 1.001799 [0.960790–1.047604] |
| complete | float64 | 300 | 0.212825 [0.203156–0.217434] | 0.206208 [0.202794–0.216403] | 1.001304 [0.983467–1.065671] |
| available | float32 | 300 | 0.101467 [0.098909–0.107800] | 0.100911 [0.099169–0.104925] | 1.001232 [0.986728–1.034729] |
| available | float64 | 300 | 0.104924 [0.103382–0.106145] | 0.103568 [0.102278–0.106397] | 1.003302 [0.977794–1.030050] |
| complete | float32 | 1000 | 0.464369 [0.458873–0.482779] | 0.462670 [0.457829–0.470997] | 1.005154 [0.985927–1.035582] |
| complete | float64 | 1000 | 0.648145 [0.633253–0.673632] | 0.648827 [0.638384–0.687796] | 1.005492 [0.920699–1.015290] |
| available | float32 | 1000 | 0.334479 [0.325178–0.343136] | 0.335668 [0.326638–0.345292] | 1.002425 [0.972234–1.022249] |
| available | float64 | 1000 | 0.338941 [0.333354–0.342820] | 0.339526 [0.335636–0.345312] | 0.992922 [0.975434–1.011507] |
| complete | float32 | 3000 | 1.258400 [1.211330–1.271773] | 1.252833 [1.194808–1.273525] | 0.998484 [0.964447–1.046932] |
| complete | float64 | 3000 | 1.727656 [1.610920–1.762942] | 1.908177 [1.837936–1.974259] | 0.899054 [0.857299–0.922452] |
| available | float32 | 3000 | 1.036028 [1.009211–1.053454] | 1.030613 [1.014304–1.050126] | 1.002993 [0.985469–1.022083] |
| available | float64 | 3000 | 1.054961 [1.024305–1.065130] | 1.051683 [1.032420–1.128133] | 0.991647 [0.941172–1.010648] |

## Complete float64: ordered transform calls at 3,000 queries

| Run | Call | Baseline seconds | Candidate seconds | Paired ratio |
| --- | --- | --- | --- | --- |
| Before/after | First | 1.642931 [1.617858–1.782201] | 1.639170 [1.608495–1.754653] | 1.002294 [0.966339–1.047102] |
| Before/after | Second | 1.825020 [1.783526–1.916701] | 1.818588 [1.768130–1.913971] | 1.003536 [0.975790–1.035135] |
| Before/after | Third | 1.458589 [1.416503–1.561179] | 1.814034 [1.768075–1.926573] | 0.801715 [0.768165–0.839483] |
| Before/after | Additional-call worker median | 1.639680 [1.606179–1.738940] | 1.816311 [1.768103–1.920272] | 0.903923 [0.871637–0.937097] |
| Same-code control | First | 1.746078 [1.626365–1.774679] | 1.750756 [1.659946–1.804913] | 0.994613 [0.951731–1.011147] |
| Same-code control | Second | 1.906870 [1.795374–1.952773] | 1.907713 [1.832480–1.973020] | 0.994645 [0.955337–1.018371] |
| Same-code control | Third | 1.548441 [1.426465–1.573112] | 1.913079 [1.843392–1.975497] | 0.803839 [0.759764–0.837750] |
| Same-code control | Additional-call worker median | 1.727656 [1.610920–1.762942] | 1.908177 [1.837936–1.974259] | 0.899054 [0.857299–0.922452] |

## Worker peak RSS

| Run | Policy | dtype | Queries | Baseline MiB | Candidate MiB | Paired delta MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Before/after | complete | float32 | 300 | 150.293 [150.141–150.445] | 150.270 [150.000–150.496] | 0.008 [-0.316–0.242] |
| Before/after | complete | float64 | 300 | 152.805 [152.559–152.945] | 152.785 [152.598–153.047] | -0.090 [-0.207–0.246] |
| Before/after | available | float32 | 300 | 248.996 [248.824–249.242] | 248.918 [248.859–249.188] | -0.039 [-0.332–0.156] |
| Before/after | available | float64 | 300 | 251.754 [251.574–252.070] | 252.051 [251.922–252.230] | 0.219 [-0.125–0.566] |
| Before/after | complete | float32 | 1000 | 150.270 [150.039–150.500] | 150.219 [149.824–150.383] | -0.113 [-0.445–0.184] |
| Before/after | complete | float64 | 1000 | 152.777 [152.645–153.156] | 152.926 [152.555–153.266] | 0.195 [-0.371–0.570] |
| Before/after | available | float32 | 1000 | 249.281 [249.152–249.363] | 249.223 [249.098–249.449] | -0.016 [-0.215–0.297] |
| Before/after | available | float64 | 1000 | 252.699 [252.484–252.723] | 252.676 [252.488–252.879] | -0.023 [-0.172–0.164] |
| Before/after | complete | float32 | 3000 | 150.285 [150.098–150.441] | 150.219 [149.949–150.379] | -0.062 [-0.223–0.031] |
| Before/after | complete | float64 | 3000 | 154.785 [154.609–154.926] | 152.730 [152.445–153.109] | -2.027 [-2.234–-1.633] |
| Before/after | available | float32 | 3000 | 257.930 [257.691–258.020] | 257.660 [257.469–257.844] | -0.223 [-0.461–-0.074] |
| Before/after | available | float64 | 3000 | 260.918 [260.781–261.090] | 263.523 [263.191–263.598] | 2.520 [2.332–2.746] |
| Same-code control | complete | float32 | 300 | 150.320 [149.988–150.445] | 150.281 [150.133–150.383] | -0.016 [-0.262–0.359] |
| Same-code control | complete | float64 | 300 | 152.824 [152.598–153.109] | 152.762 [152.441–152.867] | -0.199 [-0.496–0.270] |
| Same-code control | available | float32 | 300 | 249.082 [248.852–249.242] | 249.039 [248.879–249.137] | -0.105 [-0.355–0.250] |
| Same-code control | available | float64 | 300 | 251.965 [251.535–252.117] | 252.090 [251.934–252.168] | 0.148 [-0.051–0.520] |
| Same-code control | complete | float32 | 1000 | 150.297 [150.051–150.395] | 150.242 [150.102–150.391] | 0.055 [-0.293–0.340] |
| Same-code control | complete | float64 | 1000 | 152.801 [152.551–152.930] | 152.812 [152.551–153.582] | 0.012 [-0.168–0.691] |
| Same-code control | available | float32 | 1000 | 249.273 [249.102–249.430] | 249.262 [249.129–249.379] | -0.004 [-0.156–0.160] |
| Same-code control | available | float64 | 1000 | 252.652 [252.359–252.785] | 252.688 [252.527–252.844] | 0.043 [-0.203–0.277] |
| Same-code control | complete | float32 | 3000 | 150.359 [150.258–150.445] | 150.230 [149.930–150.465] | -0.160 [-0.402–0.098] |
| Same-code control | complete | float64 | 3000 | 154.828 [154.621–154.922] | 152.812 [152.586–152.840] | -2.008 [-2.191–-1.809] |
| Same-code control | available | float32 | 3000 | 257.934 [257.734–258.086] | 257.758 [257.500–257.875] | -0.211 [-0.496–0.043] |
| Same-code control | available | float64 | 3000 | 260.961 [260.840–261.012] | 263.516 [263.211–263.777] | 2.562 [2.250–2.926] |

## Reconstruction error and complete donor counts

All matched Faiss baseline/candidate output SHA256 digests agree, and their recorded maximum output differences are zero in both runs. Their RMSE/MAE values are identical in these records. The table shows the candidate (also the baseline) and KNN errors separately. This is observed output and aggregate metric agreement on these inputs, not a general prediction or algorithmic equivalence claim.

| Run | Policy | dtype | Queries | Method | RMSE | MAE | Scored cells | Complete donors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before/after | complete | float32 | 300 | Faiss baseline/candidate | 0.17625394 [0.16534372–0.17646811] | 0.13261908 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Before/after | complete | float32 | 300 | KNNImputer | 0.17625394 [0.16534372–0.17646811] | 0.13261908 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Before/after | complete | float64 | 300 | Faiss baseline/candidate | 0.17625394 [0.16534372–0.17646811] | 0.13261909 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Before/after | complete | float64 | 300 | KNNImputer | 0.17625394 [0.16534372–0.17646811] | 0.13261909 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Before/after | available | float32 | 300 | Faiss baseline/candidate | 0.18067464 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041304] | 1200 [1200–1200] | 2435 [2413–2447] |
| Before/after | available | float32 | 300 | KNNImputer | 0.18067464 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Before/after | available | float64 | 300 | Faiss baseline/candidate | 0.18067463 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Before/after | available | float64 | 300 | KNNImputer | 0.18067463 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Before/after | complete | float32 | 1000 | Faiss baseline/candidate | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Before/after | complete | float32 | 1000 | KNNImputer | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Before/after | complete | float64 | 1000 | Faiss baseline/candidate | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Before/after | complete | float64 | 1000 | KNNImputer | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Before/after | available | float32 | 1000 | Faiss baseline/candidate | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Before/after | available | float32 | 1000 | KNNImputer | 0.18436959 [0.17903132–0.18752558] | 0.13835251 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Before/after | available | float64 | 1000 | Faiss baseline/candidate | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Before/after | available | float64 | 1000 | KNNImputer | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Before/after | complete | float32 | 3000 | Faiss baseline/candidate | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Before/after | complete | float32 | 3000 | KNNImputer | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Before/after | complete | float64 | 3000 | Faiss baseline/candidate | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Before/after | complete | float64 | 3000 | KNNImputer | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Before/after | available | float32 | 3000 | Faiss baseline/candidate | 0.18857802 [0.18314905–0.18903595] | 0.13964838 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Before/after | available | float32 | 3000 | KNNImputer | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Before/after | available | float64 | 3000 | Faiss baseline/candidate | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Before/after | available | float64 | 3000 | KNNImputer | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Same-code control | complete | float32 | 300 | Faiss baseline/candidate | 0.17625394 [0.16534372–0.17646811] | 0.13261908 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Same-code control | complete | float32 | 300 | KNNImputer | 0.17625394 [0.16534372–0.17646811] | 0.13261908 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 300 | Faiss baseline/candidate | 0.17625394 [0.16534372–0.17646811] | 0.13261909 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 300 | KNNImputer | 0.17625394 [0.16534372–0.17646811] | 0.13261909 [0.12336287–0.13522292] | 1200 [1200–1200] | 20000 [20000–20000] |
| Same-code control | available | float32 | 300 | Faiss baseline/candidate | 0.18067464 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041304] | 1200 [1200–1200] | 2435 [2413–2447] |
| Same-code control | available | float32 | 300 | KNNImputer | 0.18067464 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Same-code control | available | float64 | 300 | Faiss baseline/candidate | 0.18067463 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Same-code control | available | float64 | 300 | KNNImputer | 0.18067463 [0.17589431–0.18364559] | 0.13643611 [0.13062049–0.14041305] | 1200 [1200–1200] | 2435 [2413–2447] |
| Same-code control | complete | float32 | 1000 | Faiss baseline/candidate | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Same-code control | complete | float32 | 1000 | KNNImputer | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 1000 | Faiss baseline/candidate | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 1000 | KNNImputer | 0.17837947 [0.16867610–0.17889660] | 0.13495129 [0.12777808–0.13525784] | 4000 [4000–4000] | 20000 [20000–20000] |
| Same-code control | available | float32 | 1000 | Faiss baseline/candidate | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Same-code control | available | float32 | 1000 | KNNImputer | 0.18436959 [0.17903132–0.18752558] | 0.13835251 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Same-code control | available | float64 | 1000 | Faiss baseline/candidate | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Same-code control | available | float64 | 1000 | KNNImputer | 0.18436959 [0.17903132–0.18752558] | 0.13835250 [0.13454628–0.14096416] | 4000 [4000–4000] | 2435 [2413–2447] |
| Same-code control | complete | float32 | 3000 | Faiss baseline/candidate | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Same-code control | complete | float32 | 3000 | KNNImputer | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 3000 | Faiss baseline/candidate | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Same-code control | complete | float64 | 3000 | KNNImputer | 0.17999296 [0.17519926–0.18193219] | 0.13533795 [0.13020665–0.13558781] | 12000 [12000–12000] | 20000 [20000–20000] |
| Same-code control | available | float32 | 3000 | Faiss baseline/candidate | 0.18857802 [0.18314905–0.18903595] | 0.13964838 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Same-code control | available | float32 | 3000 | KNNImputer | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Same-code control | available | float64 | 3000 | Faiss baseline/candidate | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |
| Same-code control | available | float64 | 3000 | KNNImputer | 0.18857802 [0.18314905–0.18903595] | 0.13964839 [0.13567473–0.14135454] | 12000 [12000–12000] | 2435 [2413–2447] |

## Interpretation and limits

- Before/after, available float64 first transform: the three configuration-level median paired ratios range from 1.164477× to 1.172912×.
- Before/after, available float64 fit plus first transform: the three configuration-level median paired ratios range from 1.145613× to 1.164727×.
- Same-code control, available float64 first transform: the three configuration-level median paired ratios range from 0.997038× to 1.014611×.
- Same-code control, available float64 fit plus first transform: the three configuration-level median paired ratios range from 0.997814× to 1.013862×.

The ordered-call tables show the baseline third-call acceleration for complete float64 in both the change comparison and the same-code control. Thus the observed discrepancy can arise without the selected-distance code change. The experiment does not establish its underlying cause, universal absence of regressions, or an allocator explanation. Same-code memory differences likewise prevent attributing the before/after RSS delta directly to the optimization.

The control is called same-code because the intervening commit only archived benchmark evidence. The archives identify commits, distribution versions and wheel hashes, but do not contain installed package source files; this analyzer does not cryptographically verify equality of package source. No cross-run timings are pooled or subtracted to correct the measured speedups.

## Reproduction

Use the [analysis workflow](https://github.com/ScionKim/FaissImputer/blob/main/.github/workflows/analyze-available-selected-distances.yml) from GitHub Actions → Run workflow. It reads preserved archives and uploads regenerated Markdown and JSON; it does not execute imputers or new timing measurements.

The [stdlib analysis script](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/analyze_available_selected_distances.py) also supports `python benchmarks/analyze_available_selected_distances.py`. The [unrounded summary](https://github.com/ScionKim/FaissImputer/blob/main/benchmarks/results/available-selected-distances-c02b71d-summary.json) retains original float values, per-record source indexes, input fingerprints, paired ratios, ordered timings, build provenance, and archive/member SHA256 digests. JSON floats use Python's round-trip representation; rounding is applied only in this Markdown report.
