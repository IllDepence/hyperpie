### reproducibility indicators

* in-text
    * Σ (\#vpa\_rels, \#vp\_rels, \#pa\_rels)
* external
    * repo stars
    * log(repo stars)
    * open issues
    * forks


### including papers w/o repo

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.12643221547893327
p-value: 0.014682015957932993
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: -0.024088932939845684
p-value: 0.6432825752285509
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.08509533247133046
p-value: 0.1012738110815795
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.09769429201484797
p-value: 0.059778596904136914
```


### only papers w/ repos

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.1276314594402173
p-value: 0.014281880493775576
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: -0.015581905551956179
p-value: 0.7657688450081295
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.08611268915815351
p-value: 0.0990712706884345
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.0988698247213978
p-value: 0.058113001390065123
```


### repos w/ at least one star

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.1301893565881363
p-value: 0.01465506065437896
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: -0.02290423450041321
p-value: 0.6689160644818567
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.08769160063827405
p-value: 0.10096311542597368
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.10066267064150931
p-value: 0.05956833429550635
```


### repos w/ at least 10 stars

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.17264774200742194
p-value: 0.006526929017364581
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: 0.03764376303407844
p-value: 0.5559780605021017
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.11769549352341689
p-value: 0.06478037953293926
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.13611061934879926
p-value: 0.032495957163403026
```


### repos w/ at least 25 stars

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.22470615904036265
p-value: 0.003929704709237551
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: 0.07987221876661862
p-value: 0.3108153464391956
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.15295640051432896
p-value: 0.051262744952950134
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.17903718838864147
p-value: 0.022212689136735714
```


### repos w/ at least 50 stars

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.31548740015799287
p-value: 0.0005938410909661382
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: 0.2304525737380974
p-value: 0.0132203471570064
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.21839992784786025
p-value: 0.019033485731513065
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.2573123475290634
p-value: 0.005501027399249911
```


### repos w/ at least 100 stars

```
n_vpa <-> repo_stars
Pearson correlation coefficient: 0.3547994315093609
p-value: 0.0015449202124055018
n_vpa <-> log(repo_stars)
Pearson correlation coefficient: 0.2871241863817553
p-value: 0.01134630398935271
n_vpa <-> repo_open_issues
Pearson correlation coefficient: 0.24274256094343036
p-value: 0.03340816559709182
n_vpa <-> repo_forks
Pearson correlation coefficient: 0.28807508571612295
p-value: 0.011066021245963086
```
