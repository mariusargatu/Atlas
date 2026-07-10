# Honest benchmark study (the regression that wasn't)

Two Atlas model versions over the answer golden set, the SAME 100 items, so the
comparison is paired. The point estimates say v_a beat v_b by three points. The
statistics say you cannot tell yet.

## Marginal scores, each with its interval
```
v_a  84/100  rate 84.0%  Wilson 95% CI [0.756, 0.899]
v_b  81/100  rate 81.0%  Wilson 95% CI [0.722, 0.875]
intervals overlap: True
```

## The paired comparison (the test that belongs on paired data)
```
discordant pairs: v_a pass / v_b fail = 7, v_a fail / v_b pass = 4
paired difference (mean v_a - mean v_b): +0.030
paired bootstrap 95% CI on the difference: [-0.030, 0.100]  (excludes zero: False)
paired permutation p: 0.553
exact McNemar p: 0.549
```

seed: 0xbeac04   resamples: 10000

## The release gate (gate on the floor, never the point)
```
candidate v_b  point 81.0%  floor 0.722  bar 0.80  budget 0.20
verdict: FAIL  (lower bound 0.722 is below the 0.800 bar: the floor has not cleared, so the gate fails closed)
```

The candidate's best guess sits above the bar and its honest floor sits below it;
shipping on the best guess is shipping on optimism, so the gate fails closed.

## The verdict

**no regression: the gap sits inside the noise, you cannot ship "v_a is better".**

The difference interval contains zero and the paired test returns p well above 0.05.
There is no regression here to find, only a smaller sample than the question deserved.
Gate a release on the lower bound of the interval, never the point.
