# Raw FINRA / SEC source archive

Source documents for the FINRA Rule 2210 training dataset. All content is archived verbatim as data. `.md` files carry a 4-line header (Title, Source URL, Retrieved, Retrieval method) followed by text extracted with the Claude built-in browser (finra.org returns 403 to curl). PDFs are the official notice PDFs; XML files are eCFR section XML.

Verification (2026-10-14): every `.md` has the 4-line header and non-trivial body text with no truncation at the end; every `.pdf` begins with `%PDF`; both `.xml` files parse as XML and contain the CFR section text (no error pages). Short files (rule_2010, rule_2020, rule_4511) are complete; those rules are one to three sentences long.

| filename | title | URL | retrieved | size (bytes) | status |
|---|---|---|---|---|---|
| rule_2010.md | 2010. STANDARDS OF COMMERCIAL HONOR AND PRINCIPLES OF TRADE \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2010 | 2026-10-14 | 1479 | ok |
| rule_2020.md | 2020. Use of Manipulative, Deceptive or Other Fraudulent Devices \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2020 | 2026-10-14 | 953 | ok |
| rule_2111.md | 2111. Suitability \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2111 | 2026-10-14 | 8605 | ok |
| rule_2210.md | 2210. Communications with the Public \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2210 | 2026-10-14 | 36500 | ok |
| rule_2211.md | 2211. Communications with the Public About Variable Life Insurance and Variable Annuities \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2211 | 2026-10-14 | 9338 | ok |
| rule_2212.md | 2212. Use of Investment Companies Rankings in Retail Communications \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2212 | 2026-10-14 | 7146 | ok |
| rule_2213.md | 2213. Requirements for the Use of Bond Mutual Fund Volatility Ratings \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2213 | 2026-10-14 | 3519 | ok |
| rule_2214.md | 2214. Requirements for the Use of Investment Analysis Tools \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2214 | 2026-10-14 | 8096 | ok |
| rule_2215.md | 2215. Communications with the Public Regarding Security Futures \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2215 | 2026-10-14 | 8494 | ok |
| rule_2216.md | 2216. Communications with the Public About Collateralized Mortgage Obligations (CMOs) \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2216 | 2026-10-14 | 6720 | ok |
| rule_2220.md | 2220. Options Communications \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2220 | 2026-10-14 | 13019 | ok |
| rule_2241.md | 2241. Research Analysts and Research Reports \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2241 | 2026-10-14 | 37044 | ok |
| rule_2330.md | 2330. Members' Responsibilities Regarding Deferred Variable Annuities \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/2330 | 2026-10-14 | 14658 | ok |
| rule_3110.md | 3110. Supervision \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/3110 | 2026-10-14 | 42771 | ok |
| rule_3240.md | 3240. Prohibition on Borrowing From or Lending to Customers \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/3240 | 2026-10-14 | 4284 | ok |
| rule_4511.md | 4511. General Requirements \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/4511 | 2026-10-14 | 985 | ok |
| rule_4512.md | 4512. Customer Account Information \| FINRA.org | https://www.finra.org/rules-guidance/rulebooks/finra-rules/4512 | 2026-10-14 | 7812 | ok |
| notice_12-29.md | Regulatory Notice 12-29 \| FINRA.org | https://www.finra.org/rules-guidance/notices/12-29 | 2026-10-14 | 74063 | ok |
| notice_12-29.pdf | Regulatory Notice 12-29 \| FINRA.org (PDF) | https://www.finra.org/sites/default/files/NoticeDocument/p127014.pdf | 2026-10-14 | 220807 | ok |
| notice_15-50.md | Regulatory Notice 15-50 \| FINRA.org — SEC Approves Rule Requiring Members' Websites to Include a Readily Apparent Reference and Hyperlink to BrokerCheck | https://www.finra.org/rules-guidance/notices/15-50 | 2026-10-14 | 8653 | ok |
| notice_15-50.pdf | Regulatory Notice 15-50 \| FINRA.org — SEC Approves Rule Requiring Members' Websites to Include a Readily Apparent Reference and Hyperlink to BrokerCheck (PDF) | https://www.finra.org/sites/default/files/Notice_Regulatory_15-50.pdf | 2026-10-14 | 75319 | ok |
| notice_16-41.md | Regulatory Notice 16-41 \| FINRA.org — SEC Approves Amendments to Rules Governing Communications With the Public | https://www.finra.org/rules-guidance/notices/16-41 | 2026-10-14 | 24511 | ok |
| notice_16-41.pdf | Regulatory Notice 16-41 \| FINRA.org — SEC Approves Amendments to Rules Governing Communications With the Public (PDF) | https://www.finra.org/sites/default/files/notice_doc_file_ref/Regulatory-Notice-16-41.pdf | 2026-10-14 | 102567 | ok |
| notice_17-18.md | Regulatory Notice 17-18 \| FINRA.org — Guidance on Social Networking Websites and Business Communications | https://www.finra.org/rules-guidance/notices/17-18 | 2026-10-14 | 22311 | ok |
| notice_17-18.pdf | Regulatory Notice 17-18 \| FINRA.org — Guidance on Social Networking Websites and Business Communications (PDF) | https://www.finra.org/sites/default/files/notice_doc_file_ref/Regulatory-Notice-17-18.pdf | 2026-10-14 | 133926 | ok |
| notice_19-31.md | Regulatory Notice 19-31 \| FINRA.org — Disclosure Innovations in Advertising and Other Communications with the Public | https://www.finra.org/rules-guidance/notices/19-31 | 2026-10-14 | 11974 | ok |
| notice_19-31.pdf | Regulatory Notice 19-31 \| FINRA.org — Disclosure Innovations in Advertising and Other Communications with the Public (PDF) | https://www.finra.org/sites/default/files/2019-09/Regulatory-Notice-19-31.pdf | 2026-10-14 | 475238 | ok |
| notice_19-32.md | Regulatory Notice 19-32 \| FINRA.org — FINRA Amends Rules 2210 and 2241 to Conform to the Fair Access to Investment Research Act of 2017 | https://www.finra.org/rules-guidance/notices/19-32 | 2026-10-14 | 21945 | ok |
| notice_19-32.pdf | Regulatory Notice 19-32 \| FINRA.org — FINRA Amends Rules 2210 and 2241 to Conform to the Fair Access to Investment Research Act of 2017 (PDF) | https://www.finra.org/sites/default/files/2019-09/Regulatory-Notice-19-32.pdf | 2026-10-14 | 133234 | ok |
| notice_20-21.md | Regulatory Notice 20-21 \| FINRA.org — FINRA Provides Guidance on Retail Communications Concerning Private Placement Offerings | https://www.finra.org/rules-guidance/notices/20-21 | 2026-10-14 | 20245 | ok |
| notice_20-21.pdf | Regulatory Notice 20-21 \| FINRA.org — FINRA Provides Guidance on Retail Communications Concerning Private Placement Offerings (PDF) | https://www.finra.org/sites/default/files/2020-06/Regulatory-Notice-20-21.pdf | 2026-10-14 | 99804 | ok |
| notice_24-09.md | Regulatory Notice 24-09 \| FINRA.org — FINRA Reminds Members of Regulatory Obligations When Using Generative Artificial Intelligence and Large Language Models | https://www.finra.org/rules-guidance/notices/24-09 | 2026-10-14 | 9558 | ok |
| notice_24-09.pdf | Regulatory Notice 24-09 \| FINRA.org — FINRA Reminds Members of Regulatory Obligations When Using Generative Artificial Intelligence and Large Language Models (PDF) | https://www.finra.org/sites/default/files/2024-06/regulatory-notice-24-09.pdf | 2026-10-14 | 133373 | ok |
| sec_rule_34b-1.xml | § 270.34b-1 Sales literature deemed to be misleading. | https://www.ecfr.gov/current/title-17/section-270.34b-1 (eCFR XML; exact API endpoint not recorded by prior agent) | 2026-10-14 | 7526 | ok |
| sec_rule_482.xml | § 230.482 Advertising by an investment company as satisfying requirements of section 10. | https://www.ecfr.gov/current/title-17/section-230.482 (eCFR XML; exact API endpoint not recorded by prior agent) | 2026-10-14 | 22862 | ok |
| faq_advertising_regulation.md | FINRA Rule 2210 Frequently Asked Questions \| FINRA.org | https://www.finra.org/rules-guidance/guidance/faqs/advertising-regulation | 2026-10-14 | 57214 | ok (accordion answers expanded before extraction) |
| keytopic_advertising_regulation.md | Advertising Regulation \| FINRA.org | https://www.finra.org/rules-guidance/key-topics/advertising-regulation | 2026-10-14 | 7491 | ok (key-topics URL redirects to /about/how-we-operate/advertising-regulation) |
| oversight_report_communications.md | Communications with the Public \| FINRA.org — 2025 FINRA Annual Regulatory Oversight Report | https://www.finra.org/rules-guidance/guidance/reports/2025-finra-annual-regulatory-oversight-report/communication-with-public | 2026-10-14 | 6882 | ok (substitute: requested communications-and-sales slug 404 for 2025/2026/2024; saved 2025 report's communication-with-public section) |

## How these map to the dataset

- **rule_2210** -> the rule digest and the violation categories in `src/finra2210/taxonomy.py`.
- **Related rules** (rule_2111, rule_2211 to rule_2216, rule_2220, rule_2241, rule_2330, rule_3110, rule_3240, rule_4511, rule_4512, rule_2010, rule_2020) -> the RELATED RULES block and the OOD `regulatory_twist` axis.
- **notice_17-18 and notice_24-09** -> the OOD channel axis (chatbots, texting, social) and the `third_party_content_adoption` category.
- **notice_12-29** -> the correspondence / retail / institutional communication definitions.
- **sec_rule_482 and sec_rule_34b-1** -> `fund_performance_disclosure`.
- Other notices (15-50, 16-41, 19-31, 19-32, 20-21), the Rule 2210 FAQ, the Advertising Regulation key-topic page and the 2025 oversight report section are supporting interpretive context; they are not mapped to a specific dataset component.
