# Folktables ACSIncome Result Interpretation

This worked example asks:

> If we only report people by age band, education band, and sex, is that coarse
> grouping enough to know the overall ACSIncome target rate? Or could the answer
> change if the hidden mix inside those groups changed?

The short answer:

> The coarse public grouping is not perfectly adequate, but the instability is
> fairly small in this sample.

## What Was Estimated

The observed target rate is `0.1237`, meaning about `12.37%` of the sampled
people exceed the ACSIncome income threshold.

The stress test keeps the public distribution fixed. In other words, it keeps
the same proportions of people in each:

```text
AGE_BAND x EDU_BAND x SEX
```

group, but allows the hidden composition inside those public groups to change.
The hidden composition includes:

- occupation major group
- class of worker
- weekly-hours band
- race
- marital status
- birthplace
- relationship status

Under that stress test, the target rate could range from:

```text
11.79% to 13.44%
```

The observed value, `12.37%`, is inside that possible range.

## Transport Ambiguity

The width of the range is:

```text
13.44% - 11.79% = 1.65 percentage points
```

That width is the transport ambiguity. It means:

> Even if age, education, and sex proportions stayed exactly the same, hidden
> composition changes could move the aggregate income-threshold rate by up to
> about 1.65 percentage points.

This is not a confidence interval. It is not saying that the estimate is
statistically uncertain by this much. It is saying:

> Given this coarse public representation, here is how much the answer could
> change if hidden subgroups inside the public cells were rearranged.

So when the report says:

```text
Public adequate: no
```

it means:

> Age band, education band, and sex alone do not fully determine the target rate
> under the chosen stress test.

At least one public group contains hidden subgroups with different
income-threshold rates.

## Worst Public Fibers

A public fiber is one coarse group, such as:

```text
under_25 x hs_or_some_college x SEX=1
```

Inside that public group, there are 7 retained hidden cells. Those hidden cells
differ by occupation, class of worker, weekly hours, race, marital status,
birthplace, and relationship status.

For that group:

```text
mass = 0.3019
range = 0.0385
contribution = 0.0116
```

Plain English:

> About 30.19% of the retained sample is in this public group. Inside it, hidden
> subgroups have target rates ranging from 0.00% to 3.85%. Because the group is
> large, this hidden variation contributes about 1.16 percentage points of the
> total 1.65 percentage-point ambiguity.

That first public group explains most of the instability.

The second meaningful contributor is:

```text
AGE_BAND=45_54, EDU_BAND=hs_or_some_college, SEX=2
```

It has:

```text
mass = 0.0450
range = 0.1096
contribution = 0.0049
```

Plain English:

> This group is smaller, only about 4.5% of the retained sample, but its hidden
> cells differ more sharply: the target rate ranges from 37.04% to 48.00%. That
> contributes about 0.49 percentage points of ambiguity.

Together, those two groups account for essentially all the transport ambiguity.
The remaining listed groups have only one retained hidden cell each, so their
range is zero. They do not add ambiguity under this stress test.

## Refinement Table

The refinement table asks:

> If we were allowed to add one hidden variable to the public representation,
> which one would make the public grouping more stable?

The best answer is:

```text
add OCC_MAJOR
```

Adding occupation major group reduces ambiguity from:

```text
0.0165 to 0.0090
```

So it removes about:

```text
0.0075 = 0.75 percentage points
```

of ambiguity.

Plain English:

> Occupation is the most valuable extra public variable. It explains the largest
> part of the hidden instability that age, education, and sex leave unresolved.

The next best refinements are:

```text
RELP        relationship status
WKHP_BAND   weekly-hours band
RAC1P       race
```

while `COW`, `MAR`, and `POBP` do not help in this particular retained state
space.

## Takeaway

With only age band, education band, and sex, the public categories are not fully
stable for estimating the ACSIncome target rate. But in this sampled, filtered
demo, the residual ambiguity is modest: about 1.65 percentage points. Most of
that ambiguity comes from one large young / less-educated public group, and
adding occupation would reduce the instability the most.

A concise README interpretation:

> In this ACSIncome sample, coarse demographic categories almost determine the
> aggregate income-threshold rate, but not quite: hidden occupational and
> household-composition differences can move the result by up to 1.65 percentage
> points even when the public demographic mix is held fixed.
