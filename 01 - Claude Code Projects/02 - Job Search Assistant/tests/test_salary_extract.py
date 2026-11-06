import pytest

from jobhunt.core.salary_extract import extract_salary


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The salary range for this role is $120,000 - $150,000.", (120000, 150000, "USD", "year")),
        ("Compensation: $120K–$150K per year plus equity", (120000, 150000, "USD", "year")),
        ("Base pay: $120-150K", (120000, 150000, "USD", "year")),
        ("Pay range $95,000 to $110,500 annually", (95000, 110500, "USD", "year")),
        ("This role pays $45 - $60/hour.", (45, 60, "USD", "hour")),
        ("Hourly rate: $22.50–$28.00 per hour", (22.5, 28, "USD", "hour")),
        ("USD 130,000 – 160,000", (130000, 160000, "USD", "year")),
        ("£55,000 - £65,000 per annum", (55000, 65000, "GBP", "year")),
        ("CA$90,000 - CA$110,000", (90000, 110000, "CAD", "year")),
        ("The starting salary for this role is $140,000.", (140000, None, "USD", "year")),
        ("Remote or NYC, $160-210k.", (160000, 210000, "USD", "year")),
        ("The salary range is between $160,000 and $210,000.", (160000, 210000, "USD", "year")),
        ("Salary: 160,000 - 210,000 USD", (160000, 210000, "USD", "year")),
        ("Pay: 160k-210k", (160000, 210000, "USD", "year")),
        ("Hourly pay: 25 - 32 per hour", (25, 32, "USD", "hour")),
    ],
)
def test_extracts_pay(text, expected):
    s = extract_salary(text)
    assert (s["salary_min"], s["salary_max"], s["salary_currency"], s["salary_period"]) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "We raised $10M - $20M in our Series A.",
        "Includes a $5,000 signing bonus and a $1,500 learning stipend.",
        "Serving 40 - 60 agencies nationwide.",
        "We have $40 - $60 per team for snacks.",
        "Our customers saved $200,000 last year.",
        "Pay grades 3-4 on the federal scale.",
        "We grew from 20 and 40 people to 200 to 400.",
    ],
)
def test_ignores_non_pay(text):
    assert extract_salary(text) is None
