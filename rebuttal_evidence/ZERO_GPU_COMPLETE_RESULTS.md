# Complete Zero-GPU Reviewer Analysis

Generated: 2026-11-15T11:10:19

This document consolidates analyses performed from existing Study 1 and Tier 1 outputs. No new model training is represented here.

## Evidence base

- Study 1: 3,200 runs
- Study 1 Adaptive observations with saved alpha: 800
- Study 1 aggregate analysis: 80 dataset x model x horizon cells
- Tier 1 / Study 2: 1,200 runs
- Adaptive vs frozen-alpha=0.5 matched comparison: 240 pairs
- Study 1 Fixed alpha=1 absolute-accuracy table: 800 runs / 80 cells
- Comparable published Autoformer check: 16 cells

## Headline findings

### Adaptive vs frozen alpha=0.5
- Mean paired MSE improvement: +0.0946%
- Median paired MSE improvement: +0.0186%
- Bootstrap 95% CI of mean MSE improvement: [-0.9565%, +1.1247%]
- MSE TOST: equivalent within +/-2% (p=0.00226), not within +/-1%
- MAE TOST: equivalent within +/-1% (p=0.00208)

### 80-cell alpha movement vs Adaptive-Fixed MSE gap
- Pearson r=0.328256, p=0.00295
- Spearman rho=0.373910, p=0.000634
- Dataset x model cluster-bootstrap 95% CI: [0.150466, 0.505629]
- Removing Exchange Rate: Pearson r=0.412693, p=0.000702
- Without Exchange Rate cluster-bootstrap 95% CI: [0.166211, 0.633293]
- Leave-one-dataset-out Pearson correlations remain positive for every dataset removal

### Tier 1 context
- lr10 vs frozen05: +4.810% mean MSE improvement; 85.42% win rate
- lr100 vs frozen05: +11.995% mean MSE improvement; 88.75% win rate
- init012 vs frozen05: -37.487% mean MSE improvement
- init088 vs frozen05: +24.523% mean MSE improvement

### Absolute-accuracy check
- Fixed alpha=1: 800 runs / 80 dataset-model-horizon cells
- Published Autoformer comparison: 16 cells over Electricity, Exchange Rate, Traffic and Weather
- Electricity, Traffic and Weather Fixed MSE is in the same regime as, and generally below, the published Autoformer reference.
- Exchange Rate is horizon-dependent rather than showing a uniform implementation deficit.

## Adaptive vs frozen05 paired analysis

Source: `zero_gpu_adaptive_vs_frozen05_report.txt`

```text
==============================================================================
STUDY 1 ADAPTIVE vs MATCHED FROZEN ALPHA=0.5
==============================================================================
Matched pairs: 240

OVERALL
MSE mean improvement   : 0.0946%
MSE median improvement : 0.0186%
MSE win rate           : 75.00%
MAE mean improvement   : 0.1430%
MAE median improvement : 0.0140%
MAE win rate           : 73.33%

MEAN RAW ERRORS
Adaptive MSE : 1.92757685
Frozen05 MSE : 1.92184720
Adaptive MAE : 0.90468164
Frozen05 MAE : 0.90425263

PAIRED TESTS
MSE paired t-test: t=0.971448, p=0.3323073
MSE Wilcoxon: W=10487.000000, p=0.00022417063

BY DATASET
                    n    mean  median  win_rate
dataset                                        
exchange_rate     120  0.2222  1.0136      60.0
national_illness  120 -0.0329  0.0133      90.0

BY MODEL
               n    mean  median  win_rate
model                                     
AutoformerMS  80 -0.0833  0.0183     61.25
PatchTST      80  1.2004  0.0110     90.00
iTransformer  80 -0.8332  0.0258     73.75

BY DATASET / MODEL
                                n    mean  median  win_rate
dataset          model                                     
exchange_rate    AutoformerMS  40 -0.0325  0.0822      52.5
                 PatchTST      40  2.3926  2.2182      80.0
                 iTransformer  40 -1.6936 -1.5090      47.5
national_illness AutoformerMS  40 -0.1341  0.0154      70.0
                 PatchTST      40  0.0082  0.0075     100.0
                 iTransformer  40  0.0271  0.0263     100.0

BY DATASET / HORIZON
                           n    mean  median  win_rate
dataset          horizon                              
exchange_rate    96       30  2.4794  1.9569   66.6667
                 192      30  2.5697  1.9628   70.0000
                 336      30  2.9514  1.7565   70.0000
                 720      30 -7.1118 -2.9314   33.3333
national_illness 24       30 -0.0149  0.0097   80.0000
                 32       30 -0.0992  0.0250   96.6667
                 48       30 -0.0143  0.0144   86.6667
                 64       30 -0.0031  0.0176   96.6667

Bootstrap 95% CI mean mse_imp_pct: [-0.9565%, 1.1247%]

Bootstrap 95% CI mean mae_imp_pct: [-0.3689%, 0.6719%]

WROTE:
zero_gpu_adaptive_vs_frozen05_pairs.csv
```

## TOST practical-equivalence analysis

Source: `zero_gpu_tost_report.txt`

```text
MSE: equivalence margin +/-1%
N: 240
Mean log gain: 0.00438456
Approximate multiplicative gain: 0.4394%
90% CI as percentage gain: [-0.4489%, 1.3357%]
Lower-bound test p: 0.0041176694
Upper-bound test p: 0.15096485
TOST p-value: 0.15096485
Equivalent: NO

MSE: equivalence margin +/-2%
N: 240
Mean log gain: 0.00438456
Approximate multiplicative gain: 0.4394%
90% CI as percentage gain: [-0.4489%, 1.3357%]
Lower-bound test p: 5.4052683e-06
Upper-bound test p: 0.0022648735
TOST p-value: 0.0022648735
Equivalent: YES

MAE: equivalence margin +/-1%
N: 240
Mean log gain: 0.00226945
Approximate multiplicative gain: 0.2272%
90% CI as percentage gain: [-0.2113%, 0.6676%]
Lower-bound test p: 3.3916213e-06
Upper-bound test p: 0.0020846876
TOST p-value: 0.0020846876
Equivalent: YES

MAE: equivalence margin +/-2%
N: 240
Mean log gain: 0.00226945
Approximate multiplicative gain: 0.2272%
90% CI as percentage gain: [-0.2113%, 0.6676%]
Lower-bound test p: 3.5527137e-15
Upper-bound test p: 1.2873904e-10
TOST p-value: 1.2873904e-10
Equivalent: YES
```

## 80-cell correlation analysis

Source: `zero_gpu_80cell_report.txt`

```text
==============================================================================
80-CELL ALPHA MOVEMENT VS ADAPTIVE-FIXED GAP
==============================================================================
Input rows: 800
Cells: 80
Seeds per cell: [np.int64(10)]

------------------------------------------------------------------------------
ALL 80 CELLS
------------------------------------------------------------------------------
N CELLS: 80

X = mean_abs_alpha_shift

Y = mse_gain_pct
Pearson  r=0.328256, p=0.00295252, bootstrap95=[0.229537, 0.442369]
Spearman r=0.373910, p=0.000634231, bootstrap95=[0.176375, 0.542899]

Y = mae_gain_pct
Pearson  r=0.311909, p=0.00485471, bootstrap95=[0.203379, 0.427860]
Spearman r=0.270136, p=0.0153768, bootstrap95=[0.070194, 0.451100]

X = mean_alpha_shift

Y = mse_gain_pct
Pearson  r=0.195906, p=0.0815818, bootstrap95=[0.069990, 0.330515]
Spearman r=0.272738, p=0.0143793, bootstrap95=[0.050752, 0.468687]

Y = mae_gain_pct
Pearson  r=0.091505, p=0.419518, bootstrap95=[-0.054737, 0.244060]
Spearman r=-0.062049, p=0.584534, bootstrap95=[-0.283105, 0.174079]

------------------------------------------------------------------------------
WITHOUT EXCHANGE RATE
------------------------------------------------------------------------------
N CELLS: 64

X = mean_abs_alpha_shift

Y = mse_gain_pct
Pearson  r=0.412693, p=0.000701794, bootstrap95=[0.296963, 0.539432]
Spearman r=0.458974, p=0.000136421, bootstrap95=[0.234409, 0.632654]

Y = mae_gain_pct
Pearson  r=0.318629, p=0.0102863, bootstrap95=[0.167379, 0.463843]
Spearman r=0.280220, p=0.0249152, bootstrap95=[0.026982, 0.491686]

X = mean_alpha_shift

Y = mse_gain_pct
Pearson  r=0.298189, p=0.0167014, bootstrap95=[0.178063, 0.427423]
Spearman r=0.370559, p=0.00257713, bootstrap95=[0.140883, 0.565654]

Y = mae_gain_pct
Pearson  r=0.072841, p=0.567322, bootstrap95=[-0.102462, 0.252134]
Spearman r=-0.103114, p=0.417472, bootstrap95=[-0.339726, 0.162718]

==============================================================================
DATASET DESCRIPTIVES
==============================================================================
                  cells  mean_abs_alpha_shift  mean_mse_gain_pct  median_mse_gain_pct  mse_win_rate
dataset                                                                                            
electricity          16              0.037913         -10.156837            -6.321877          25.0
exchange_rate        16              0.004453         -69.905829           -62.710827          12.5
national_illness     16              0.000641         -22.626412           -19.939280          25.0
traffic              16              0.020231          -5.138779            -5.289017           0.0
weather              16              0.029260         -15.873138             2.367215          62.5

WROTE:
zero_gpu_80cell_data.csv
zero_gpu_80cell_dataset_summary.csv
```

## Cluster-bootstrap and leave-one-dataset-out robustness

Source: `zero_gpu_cluster_robustness_report.txt`

```text
==============================================================================
ALL 80 CELLS
==============================================================================
Cells: 80
Pearson:  r=0.328256, p=0.0029525203
Spearman: r=0.373910, p=0.00063423144

LEAVE-ONE-DATASET-OUT
remove electricity          r=0.280625
remove exchange_rate        r=0.412693
remove national_illness     r=0.380203
remove traffic              r=0.351695
remove weather              r=0.261000

DATASET x MODEL CLUSTER BOOTSTRAP
Clusters: 20
Valid bootstrap samples: 2000
95% CI: [0.150466, 0.505629]
P(r <= 0): 0.001000

==============================================================================
WITHOUT EXCHANGE RATE
==============================================================================
Cells: 64
Pearson:  r=0.412693, p=0.0007017937
Spearman: r=0.458974, p=0.00013642076

LEAVE-ONE-DATASET-OUT
remove electricity          r=0.385500
remove national_illness     r=0.407593
remove traffic              r=0.468609
remove weather              r=0.385696

DATASET x MODEL CLUSTER BOOTSTRAP
Clusters: 16
Valid bootstrap samples: 2000
95% CI: [0.166211, 0.633293]
P(r <= 0): 0.000000

WROTE:
zero_gpu_leave_one_dataset_out.csv
zero_gpu_cluster_bootstrap_ci.csv
```

## Study 1 Adaptive absolute performance

Source: `zero_gpu_absolute_table_report.txt`

```text
==============================================================================
ABSOLUTE STUDY 1 ADAPTIVE PERFORMANCE
==============================================================================
Cells: 80
Seeds per cell: [np.int64(10)]

BY DATASET / MODEL / HORIZON
         dataset        model  horizon  n  mse_mean   mse_sd  mae_mean   mae_sd
     electricity AutoformerMS       96 10  0.214661 0.004010  0.325437 0.003749
     electricity AutoformerMS      192 10  0.231171 0.004555  0.336805 0.003882
     electricity AutoformerMS      336 10  0.241572 0.005877  0.343852 0.005736
     electricity AutoformerMS      720 10  0.267337 0.009881  0.361582 0.009430
     electricity   InformerMS       96 10  0.250272 0.012752  0.355579 0.010687
     electricity   InformerMS      192 10  0.261993 0.006285  0.367215 0.004948
     electricity   InformerMS      336 10  0.288207 0.009082  0.386585 0.006955
     electricity   InformerMS      720 10  0.304349 0.008590  0.399315 0.007417
     electricity     PatchTST       96 10  0.172900 0.000960  0.262875 0.001385
     electricity     PatchTST      192 10  0.178576 0.000439  0.269775 0.000538
     electricity     PatchTST      336 10  0.193133 0.001239  0.286198 0.001003
     electricity     PatchTST      720 10  0.227735 0.001217  0.318711 0.001243
     electricity iTransformer       96 10  0.154130 0.000397  0.251103 0.000381
     electricity iTransformer      192 10  0.167178 0.000838  0.264642 0.000609
     electricity iTransformer      336 10  0.183408 0.001758  0.284907 0.001964
     electricity iTransformer      720 10  0.216655 0.002655  0.317868 0.002965
   exchange_rate AutoformerMS       96 10  0.286703 0.048607  0.399700 0.036240
   exchange_rate AutoformerMS      192 10  0.579212 0.128861  0.591095 0.074884
   exchange_rate AutoformerMS      336 10  0.793933 0.120225  0.718369 0.052687
   exchange_rate AutoformerMS      720 10  2.241867 0.622292  1.234154 0.171867
   exchange_rate   InformerMS       96 10  0.277145 0.030184  0.391002 0.026480
   exchange_rate   InformerMS      192 10  0.486386 0.039198  0.508970 0.019362
   exchange_rate   InformerMS      336 10  0.826078 0.043963  0.693173 0.016115
   exchange_rate   InformerMS      720 10  0.751798 0.136135  0.675828 0.046731
   exchange_rate     PatchTST       96 10  0.196595 0.019941  0.339798 0.017944
   exchange_rate     PatchTST      192 10  0.337588 0.026602  0.455626 0.018758
   exchange_rate     PatchTST      336 10  0.517941 0.014550  0.575800 0.009222
   exchange_rate     PatchTST      720 10  0.805617 0.072546  0.737604 0.029098
   exchange_rate iTransformer       96 10  0.282033 0.029647  0.401569 0.021970
   exchange_rate iTransformer      192 10  0.441307 0.037621  0.519849 0.021785
   exchange_rate iTransformer      336 10  0.590870 0.056998  0.612627 0.028025
   exchange_rate iTransformer      720 10  0.941435 0.136336  0.787592 0.050081
national_illness AutoformerMS       24 10  2.949098 0.143699  1.117008 0.032145
national_illness AutoformerMS       32 10  2.797107 0.177651  1.088145 0.041323
national_illness AutoformerMS       48 10  3.087110 0.247225  1.119179 0.049494
national_illness AutoformerMS       64 10  3.229437 0.233224  1.172643 0.053397
national_illness   InformerMS       24 10  4.267846 0.066472  1.323857 0.011253
national_illness   InformerMS       32 10  4.266369 0.101547  1.347591 0.020850
national_illness   InformerMS       48 10  4.364520 0.114873  1.373573 0.022691
national_illness   InformerMS       64 10  4.520489 0.141665  1.399122 0.035501
national_illness     PatchTST       24 10  2.914768 0.070521  1.124683 0.013543
national_illness     PatchTST       32 10  3.081472 0.063709  1.165347 0.012043
national_illness     PatchTST       48 10  3.054976 0.044922  1.168510 0.009836
national_illness     PatchTST       64 10  3.343728 0.046602  1.233915 0.009588
national_illness iTransformer       24 10  3.256847 0.237693  1.253994 0.041305
national_illness iTransformer       32 10  3.501169 0.203818  1.297783 0.037201
national_illness iTransformer       48 10  3.305618 0.119718  1.252800 0.026664
national_illness iTransformer       64 10  3.725413 0.089071  1.344570 0.019139
         traffic AutoformerMS       96 10  0.585515 0.011715  0.333298 0.005517
         traffic AutoformerMS      192 10  0.607467 0.010144  0.339518 0.008920
         traffic AutoformerMS      336 10  0.625416 0.009163  0.341340 0.006061
         traffic AutoformerMS      720 10  0.652340 0.010377  0.351805 0.007116
         traffic   InformerMS       96 10  0.608364 0.009977  0.376175 0.005571
         traffic   InformerMS      192 10  0.649058 0.009817  0.396448 0.004154
         traffic   InformerMS      336 10  0.719261 0.015486  0.432135 0.014074
         traffic   InformerMS      720 10  0.811613 0.028386  0.474542 0.017406
         traffic     PatchTST       96 10  0.506926 0.003234  0.317496 0.002980
         traffic     PatchTST      192 10  0.502700 0.002212  0.315255 0.002082
         traffic     PatchTST      336 10  0.514543 0.003366  0.319526 0.002918
         traffic     PatchTST      720 10  0.555770 0.004695  0.340394 0.003825
         traffic iTransformer       96 10  0.449219 0.026979  0.306237 0.019577
         traffic iTransformer      192 10  0.474384 0.026739  0.319871 0.020608
         traffic iTransformer      336 10  0.492168 0.022203  0.326923 0.017771
         traffic iTransformer      720 10  0.525532 0.017074  0.344368 0.011982
         weather AutoformerMS       96 10  0.257623 0.029016  0.346779 0.031132
         weather AutoformerMS      192 10  0.414642 0.064329  0.459368 0.043619
         weather AutoformerMS      336 10  0.527967 0.057770  0.526312 0.031894
         weather AutoformerMS      720 10  0.807681 0.116227  0.662431 0.055619
         weather   InformerMS       96 10  0.292990 0.039828  0.345613 0.028396
         weather   InformerMS      192 10  0.337859 0.018342  0.380004 0.021041
         weather   InformerMS      336 10  0.400661 0.026890  0.427360 0.014571
         weather   InformerMS      720 10  0.590479 0.041772  0.541533 0.020619
         weather     PatchTST       96 10  0.169279 0.001041  0.233164 0.001980
         weather     PatchTST      192 10  0.208499 0.001004  0.270890 0.001414
         weather     PatchTST      336 10  0.258692 0.001825  0.312388 0.003266
         weather     PatchTST      720 10  0.330788 0.005531  0.371190 0.008215
         weather iTransformer       96 10  0.171952 0.002704  0.233552 0.004647
         weather iTransformer      192 10  0.213362 0.002120  0.272489 0.002776
         weather iTransformer      336 10  0.265887 0.003047  0.315544 0.005111
         weather iTransformer      720 10  0.336495 0.003372  0.368818 0.004947

==============================================================================
POOLED ACROSS HORIZONS / SEEDS
==============================================================================
         dataset        model  runs  mse_mean   mse_sd  mae_mean   mae_sd
     electricity AutoformerMS    40  0.238685 0.020349  0.341919 0.014536
     electricity   InformerMS    40  0.276206 0.023387  0.377174 0.018689
     electricity     PatchTST    40  0.193086 0.021616  0.284390 0.021850
     electricity iTransformer    40  0.180343 0.023742  0.279630 0.025522
   exchange_rate AutoformerMS    40  0.975429 0.823803  0.735829 0.327334
   exchange_rate   InformerMS    40  0.585352 0.232633  0.567243 0.129370
   exchange_rate     PatchTST    40  0.464435 0.233709  0.527207 0.150486
   exchange_rate iTransformer    40  0.563911 0.257929  0.580409 0.146270
national_illness AutoformerMS    40  3.015688 0.255178  1.124244 0.053015
national_illness   InformerMS    40  4.354806 0.148604  1.361036 0.036773
national_illness     PatchTST    40  3.098736 0.166393  1.173114 0.041104
national_illness iTransformer    40  3.447262 0.250574  1.287287 0.049198
         traffic AutoformerMS    40  0.617684 0.026740  0.341490 0.009541
         traffic   InformerMS    40  0.697074 0.079910  0.419825 0.039522
         traffic     PatchTST    40  0.519985 0.021622  0.323168 0.010592
         traffic iTransformer    40  0.485326 0.036132  0.324350 0.022049
         weather AutoformerMS    40  0.501978 0.215496  0.498723 0.122455
         weather   InformerMS    40  0.405497 0.119213  0.423628 0.077794
         weather     PatchTST    40  0.241815 0.061195  0.296908 0.052068
         weather iTransformer    40  0.246924 0.062345  0.297601 0.051137

WROTE:
zero_gpu_absolute_all_modes.csv
zero_gpu_absolute_adaptive.csv
zero_gpu_absolute_adaptive_pooled.csv
zero_gpu_published_reference_template.csv
```

## Study 1 Fixed alpha=1 absolute performance

Source: `zero_gpu_fixed_absolute_report.txt`

```text
==============================================================================
STUDY 1 FIXED ALPHA=1 ABSOLUTE PERFORMANCE
==============================================================================
RUNS: 800
CELLS: 80
SEEDS PER CELL: [np.int64(10)]

FULL TABLE
         dataset        model  horizon  n  mse_mean   mse_sd  mae_mean   mae_sd
     electricity AutoformerMS       96 10  0.186475 0.003481  0.302101 0.003336
     electricity AutoformerMS      192 10  0.204274 0.009481  0.318641 0.007959
     electricity AutoformerMS      336 10  0.209849 0.004949  0.325902 0.005481
     electricity AutoformerMS      720 10  0.239837 0.008843  0.351336 0.007298
     electricity   InformerMS       96 10  0.188455 0.001321  0.301819 0.001673
     electricity   InformerMS      192 10  0.202969 0.002198  0.315277 0.002261
     electricity   InformerMS      336 10  0.227710 0.007813  0.339172 0.008101
     electricity   InformerMS      720 10  0.251754 0.006334  0.357167 0.005131
     electricity     PatchTST       96 10  0.170888 0.000224  0.256938 0.000295
     electricity     PatchTST      192 10  0.178660 0.000166  0.264653 0.000154
     electricity     PatchTST      336 10  0.194907 0.000292  0.281192 0.000298
     electricity     PatchTST      720 10  0.234375 0.000336  0.314072 0.000345
     electricity iTransformer       96 10  0.152483 0.000315  0.243391 0.000245
     electricity iTransformer      192 10  0.165802 0.000597  0.255743 0.000747
     electricity iTransformer      336 10  0.182763 0.001012  0.273823 0.001320
     electricity iTransformer      720 10  0.219616 0.000766  0.306079 0.000522
   exchange_rate AutoformerMS       96 10  0.139175 0.015288  0.272304 0.013951
   exchange_rate AutoformerMS      192 10  0.421721 0.167682  0.470821 0.083439
   exchange_rate AutoformerMS      336 10  0.608978 0.220492  0.588037 0.101449
   exchange_rate AutoformerMS      720 10  0.870473 0.414248  0.731046 0.156786
   exchange_rate   InformerMS       96 10  0.151948 0.045823  0.282390 0.034057
   exchange_rate   InformerMS      192 10  0.429116 0.073253  0.484092 0.044100
   exchange_rate   InformerMS      336 10  0.529639 0.061776  0.551234 0.032776
   exchange_rate   InformerMS      720 10  0.989414 0.136853  0.782619 0.044339
   exchange_rate     PatchTST       96 10  0.087211 0.002829  0.204090 0.002550
   exchange_rate     PatchTST      192 10  0.182358 0.004600  0.302165 0.002616
   exchange_rate     PatchTST      336 10  0.347014 0.025225  0.420963 0.010839
   exchange_rate     PatchTST      720 10  0.923248 0.034874  0.715173 0.010821
   exchange_rate iTransformer       96 10  0.092277 0.001767  0.214674 0.001945
   exchange_rate iTransformer      192 10  0.188038 0.005513  0.311207 0.005251
   exchange_rate iTransformer      336 10  0.348695 0.023832  0.426516 0.014279
   exchange_rate iTransformer      720 10  0.916753 0.049092  0.721713 0.017971
national_illness AutoformerMS       24 10  3.144714 0.162037  1.155504 0.038052
national_illness AutoformerMS       32 10  3.036051 0.170009  1.147694 0.051579
national_illness AutoformerMS       48 10  3.307790 0.183016  1.176414 0.037245
national_illness AutoformerMS       64 10  3.421350 0.175137  1.205825 0.027471
national_illness   InformerMS       24 10  3.809755 0.124776  1.270931 0.026139
national_illness   InformerMS       32 10  3.750477 0.155547  1.267825 0.036198
national_illness   InformerMS       48 10  3.748319 0.212867  1.257819 0.050233
national_illness   InformerMS       64 10  3.550927 0.099567  1.216690 0.026757
national_illness     PatchTST       24 10  2.361299 0.073680  0.940346 0.012271
national_illness     PatchTST       32 10  2.723154 0.062288  1.006367 0.009694
national_illness     PatchTST       48 10  2.435884 0.050349  0.965835 0.010723
national_illness     PatchTST       64 10  2.208546 0.025898  0.956820 0.006262
national_illness iTransformer       24 10  2.214141 0.074614  0.885689 0.012504
national_illness iTransformer       32 10  2.502538 0.069371  0.961100 0.006001
national_illness iTransformer       48 10  2.197479 0.026133  0.932874 0.005934
national_illness iTransformer       64 10  2.217074 0.043417  0.965015 0.011043
         traffic AutoformerMS       96 10  0.558060 0.007760  0.342123 0.004806
         traffic AutoformerMS      192 10  0.574935 0.010279  0.348410 0.006899
         traffic AutoformerMS      336 10  0.587945 0.007957  0.353283 0.006245
         traffic AutoformerMS      720 10  0.612625 0.003058  0.365631 0.002727
         traffic   InformerMS       96 10  0.582241 0.008048  0.352160 0.004243
         traffic   InformerMS      192 10  0.610433 0.009072  0.366225 0.006565
         traffic   InformerMS      336 10  0.674113 0.020622  0.401016 0.018604
         traffic   InformerMS      720 10  0.758239 0.017726  0.440069 0.012410
         traffic     PatchTST       96 10  0.485603 0.001082  0.319333 0.001822
         traffic     PatchTST      192 10  0.487712 0.001922  0.317623 0.001773
         traffic     PatchTST      336 10  0.502355 0.001739  0.323010 0.001948
         traffic     PatchTST      720 10  0.537068 0.001784  0.341265 0.002525
         traffic iTransformer       96 10  0.429450 0.002344  0.293023 0.002003
         traffic iTransformer      192 10  0.447604 0.001137  0.299851 0.001223
         traffic iTransformer      336 10  0.465806 0.000659  0.309347 0.001075
         traffic iTransformer      720 10  0.502337 0.001132  0.332634 0.001542
         weather AutoformerMS       96 10  0.183298 0.010457  0.262727 0.011685
         weather AutoformerMS      192 10  0.247737 0.015703  0.326256 0.011794
         weather AutoformerMS      336 10  0.299860 0.019296  0.371193 0.020340
         weather AutoformerMS      720 10  0.403912 0.043073  0.441954 0.024216
         weather   InformerMS       96 10  0.242181 0.023332  0.319103 0.019978
         weather   InformerMS      192 10  0.337286 0.026505  0.396520 0.018361
         weather   InformerMS      336 10  0.456819 0.050472  0.466853 0.024009
         weather   InformerMS      720 10  0.602679 0.055394  0.553535 0.024871
         weather     PatchTST       96 10  0.173994 0.000770  0.217291 0.000893
         weather     PatchTST      192 10  0.220488 0.000631  0.258218 0.000691
         weather     PatchTST      336 10  0.276697 0.000708  0.298742 0.000701
         weather     PatchTST      720 10  0.352972 0.001116  0.348241 0.000608
         weather iTransformer       96 10  0.172078 0.001611  0.211415 0.001611
         weather iTransformer      192 10  0.223334 0.001592  0.256590 0.001471
         weather iTransformer      336 10  0.281062 0.002772  0.299295 0.001775
         weather iTransformer      720 10  0.357749 0.001337  0.350000 0.001083

WROTE:
zero_gpu_fixed_absolute.csv
```

## Fixed alpha=1 vs published Autoformer

**MISSING: `zero_gpu_autoformer_vs_published.txt`**

## Machine-readable files

- `final_analysis_3200/results.csv` (479728 bytes)
- `tier1_all_runs.csv` (379293 bytes)
- `zero_gpu_adaptive_vs_frozen05_pairs.csv` (45762 bytes)
- `zero_gpu_80cell_data.csv` (19958 bytes)
- `zero_gpu_80cell_dataset_summary.csv` (481 bytes)
- `zero_gpu_leave_one_dataset_out.csv` (419 bytes)
- `zero_gpu_cluster_bootstrap_ci.csv` (152 bytes)
- `zero_gpu_absolute_adaptive.csv` (11084 bytes)
- `zero_gpu_fixed_absolute.csv` (8877 bytes)
- `zero_gpu_autoformer_vs_published.csv` (3413 bytes)