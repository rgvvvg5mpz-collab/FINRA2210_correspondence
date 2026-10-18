# Release gate: PASS

Candidate `results/metrics/fine_tuned_{split}.json` vs baseline `results/metrics/base_zero_shot_{split}.json`.

| Scope | Metric | Candidate | Baseline | Result |
|---|---|---|---|---|
| test overall | json_parse_rate | 0.998 | 0.678 | ok |
| test overall | compliant_accuracy | 0.965 | 0.392 | ok |
| test overall | violation_detection.recall | 0.973 | 0.341 | ok |
| test overall | communication_type_accuracy | 0.998 | 0.000 | ok |
| test overall | severity_accuracy | 0.895 | 0.012 | ok |
| test overall | violation_category_micro.f1 | 0.797 | 0.000 | ok |
| test overall | category_broad_group_micro.f1 | 0.921 | 0.000 | ok |
| test chat | compliant_accuracy | 0.974 | 0.384 | ok |
| test chat | violation_category_micro.f1 | 0.791 | 0.000 | ok |
| test group misleading_content | recall | 0.954 | 0.000 | ok |
| test group misleading_content | f1 | 0.957 | 0.000 | ok |
| test group required_disclosures | recall | 0.764 | 0.000 | ok |
| test group required_disclosures | f1 | 0.833 | 0.000 | ok |
| test group regulator_endorsement | recall | 0.919 | 0.000 | ok |
| test group regulator_endorsement | f1 | 0.958 | 0.000 | ok |
| test easy | compliant_accuracy | 0.966 | 0.453 | ok |
| test easy | violation_category_micro.f1 | 0.874 | 0.000 | ok |
| test hard | compliant_accuracy | 0.958 | 0.368 | ok |
| test hard | violation_category_micro.f1 | 0.759 | 0.000 | ok |
| test medium | compliant_accuracy | 0.971 | 0.365 | ok |
| test medium | violation_category_micro.f1 | 0.796 | 0.000 | ok |
| ood overall | json_parse_rate | 0.997 | 0.654 | ok |
| ood overall | compliant_accuracy | 0.953 | 0.396 | ok |
| ood overall | violation_detection.recall | 0.973 | 0.390 | ok |
| ood overall | communication_type_accuracy | 0.986 | 0.000 | ok |
| ood overall | severity_accuracy | 0.864 | 0.020 | ok |
| ood overall | violation_category_micro.f1 | 0.686 | 0.000 | ok |
| ood overall | category_broad_group_micro.f1 | 0.876 | 0.000 | ok |
| ood chat | compliant_accuracy | 0.964 | 0.433 | ok |
| ood chat | violation_category_micro.f1 | 0.693 | 0.000 | ok |
| ood group misleading_content | recall | 0.952 | 0.000 | ok |
| ood group misleading_content | f1 | 0.938 | 0.000 | ok |
| ood group required_disclosures | recall | 0.675 | 0.000 | ok |
| ood group required_disclosures | f1 | 0.725 | 0.000 | ok |
| ood group approval_and_audience | recall | 0.656 | 0.000 | ok |
| ood group approval_and_audience | f1 | 0.704 | 0.000 | ok |
| ood group regulator_endorsement | recall | 0.891 | 0.000 | ok |
| ood group regulator_endorsement | f1 | 0.922 | 0.000 | ok |
| ood easy | compliant_accuracy | 0.938 | 0.441 | ok |
| ood easy | violation_category_micro.f1 | 0.717 | 0.000 | ok |
| ood hard | compliant_accuracy | 0.961 | 0.377 | ok |
| ood hard | violation_category_micro.f1 | 0.658 | 0.000 | ok |
| ood medium | compliant_accuracy | 0.958 | 0.376 | ok |
| ood medium | violation_category_micro.f1 | 0.695 | 0.000 | ok |
