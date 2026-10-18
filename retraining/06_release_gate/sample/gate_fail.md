# Release gate: FAIL

Candidate `results/metrics/base_zero_shot_{split}.json` vs baseline `results/metrics/fine_tuned_{split}.json`.

| Scope | Metric | Candidate | Baseline | Result |
|---|---|---|---|---|
| test overall | json_parse_rate | 0.678 | 0.998 | FAIL: below floor 0.99; dropped 0.320 > 0.005 |
| test overall | compliant_accuracy | 0.392 | 0.965 | FAIL: below floor 0.94; dropped 0.573 > 0.010 |
| test overall | violation_detection.recall | 0.341 | 0.973 | FAIL: below floor 0.95; dropped 0.632 > 0.010 |
| test overall | communication_type_accuracy | 0.000 | 0.998 | FAIL: below floor 0.97; dropped 0.998 > 0.010 |
| test overall | severity_accuracy | 0.012 | 0.895 | FAIL: dropped 0.884 > 0.020 |
| test overall | violation_category_micro.f1 | 0.000 | 0.797 | FAIL: below floor 0.65; dropped 0.797 > 0.020 |
| test overall | category_broad_group_micro.f1 | 0.000 | 0.921 | FAIL: below floor 0.85; dropped 0.921 > 0.020 |
| test chat | compliant_accuracy | 0.384 | 0.974 | FAIL: below floor 0.94; dropped 0.590 > 0.010 |
| test chat | violation_category_micro.f1 | 0.000 | 0.791 | FAIL: dropped 0.791 > 0.030 |
| test group misleading_content | recall | 0.000 | 0.954 | FAIL: dropped 0.954 > 0.050 |
| test group misleading_content | f1 | 0.000 | 0.957 | FAIL: dropped 0.957 > 0.050 |
| test group required_disclosures | recall | 0.000 | 0.764 | FAIL: dropped 0.764 > 0.050 |
| test group required_disclosures | f1 | 0.000 | 0.833 | FAIL: dropped 0.833 > 0.050 |
| test group regulator_endorsement | recall | 0.000 | 0.919 | FAIL: dropped 0.919 > 0.050 |
| test group regulator_endorsement | f1 | 0.000 | 0.958 | FAIL: dropped 0.958 > 0.050 |
| test easy | compliant_accuracy | 0.453 | 0.966 | FAIL: dropped 0.513 > 0.020 |
| test easy | violation_category_micro.f1 | 0.000 | 0.874 | FAIL: dropped 0.874 > 0.040 |
| test hard | compliant_accuracy | 0.368 | 0.958 | FAIL: dropped 0.590 > 0.020 |
| test hard | violation_category_micro.f1 | 0.000 | 0.759 | FAIL: dropped 0.759 > 0.040 |
| test medium | compliant_accuracy | 0.365 | 0.971 | FAIL: dropped 0.605 > 0.020 |
| test medium | violation_category_micro.f1 | 0.000 | 0.796 | FAIL: dropped 0.796 > 0.040 |
| ood overall | json_parse_rate | 0.654 | 0.997 | FAIL: below floor 0.99; dropped 0.343 > 0.005 |
| ood overall | compliant_accuracy | 0.396 | 0.953 | FAIL: below floor 0.94; dropped 0.557 > 0.010 |
| ood overall | violation_detection.recall | 0.390 | 0.973 | FAIL: below floor 0.95; dropped 0.584 > 0.010 |
| ood overall | communication_type_accuracy | 0.000 | 0.986 | FAIL: below floor 0.97; dropped 0.986 > 0.010 |
| ood overall | severity_accuracy | 0.020 | 0.864 | FAIL: dropped 0.844 > 0.020 |
| ood overall | violation_category_micro.f1 | 0.000 | 0.686 | FAIL: below floor 0.65; dropped 0.686 > 0.020 |
| ood overall | category_broad_group_micro.f1 | 0.000 | 0.876 | FAIL: below floor 0.85; dropped 0.876 > 0.020 |
| ood chat | compliant_accuracy | 0.433 | 0.964 | FAIL: below floor 0.94; dropped 0.530 > 0.010 |
| ood chat | violation_category_micro.f1 | 0.000 | 0.693 | FAIL: dropped 0.693 > 0.030 |
| ood group misleading_content | recall | 0.000 | 0.952 | FAIL: dropped 0.952 > 0.050 |
| ood group misleading_content | f1 | 0.000 | 0.938 | FAIL: dropped 0.938 > 0.050 |
| ood group required_disclosures | recall | 0.000 | 0.675 | FAIL: dropped 0.675 > 0.050 |
| ood group required_disclosures | f1 | 0.000 | 0.725 | FAIL: dropped 0.725 > 0.050 |
| ood group approval_and_audience | recall | 0.000 | 0.656 | FAIL: dropped 0.656 > 0.050 |
| ood group approval_and_audience | f1 | 0.000 | 0.704 | FAIL: dropped 0.704 > 0.050 |
| ood group regulator_endorsement | recall | 0.000 | 0.891 | FAIL: dropped 0.891 > 0.050 |
| ood group regulator_endorsement | f1 | 0.000 | 0.922 | FAIL: dropped 0.922 > 0.050 |
| ood easy | compliant_accuracy | 0.441 | 0.938 | FAIL: dropped 0.497 > 0.020 |
| ood easy | violation_category_micro.f1 | 0.000 | 0.717 | FAIL: dropped 0.717 > 0.040 |
| ood hard | compliant_accuracy | 0.377 | 0.961 | FAIL: dropped 0.583 > 0.020 |
| ood hard | violation_category_micro.f1 | 0.000 | 0.658 | FAIL: dropped 0.658 > 0.040 |
| ood medium | compliant_accuracy | 0.376 | 0.958 | FAIL: dropped 0.582 > 0.020 |
| ood medium | violation_category_micro.f1 | 0.000 | 0.695 | FAIL: dropped 0.695 > 0.040 |
