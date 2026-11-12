"""The public demo's fictional persona and everything about them (asked 2026-11-11).

Every person, organization, board, posting and conversation here is invented. None of it comes from
the real database or the owner's documents, and no real company has postings attributed to it.
demo/seed.py turns this into a database; demo/record.py records the app's answers from it.
"""

PERSON = {
    "name": "Jordan Rivera",
    "email": "jordan.rivera@example.com",
}

# ---------------------------------------------------------------------------------------------
# Documents. Each paragraph is (style, text); seed.py writes them as real .docx files.

RESUME = [
    ("Title", "JORDAN RIVERA"),
    ("Normal", "Arlington, VA · jordan.rivera@example.com · (555) 010-4471"),
    ("Heading 1", "SUMMARY"),
    (
        "Normal",
        "Analytics leader with 12 years turning messy public-sector and nonprofit data into decisions. "
        "Builds small, senior data teams, ships measurement that program staff actually use, and "
        "translates between engineers, executives and the communities a program serves.",
    ),
    ("Heading 1", "EXPERIENCE"),
    ("job", "Common Ledger | Director of Analytics | 2021 - Present"),
    ("bullet", "Lead a team of 7 analysts and data engineers supporting 40 state and city benefit programs."),
    (
        "bullet",
        "Built the outcomes framework now used in every client contract, cutting reporting time by 60%.",
    ),
    (
        "bullet",
        "Replaced a tangle of spreadsheets with a dbt + BigQuery warehouse serving 300 internal users.",
    ),
    ("bullet", "Partner with product and policy leads to define success metrics before a program launches."),
    ("job", "Fairway Health Partners | Senior Manager, Data Science | 2017 - 2021"),
    ("bullet", "Managed 4 data scientists building enrollment forecasts for Medicaid outreach campaigns."),
    ("bullet", "Designed a randomized outreach test that raised renewal completion 11% across 3 states."),
    ("bullet", "Presented findings to state agency leadership and wrote the playbook adopted by 2 partners."),
    ("job", "Open Streets Institute | Data Analyst | 2013 - 2017"),
    ("bullet", "Built the transit-equity dashboard cited in two city council budget hearings."),
    ("bullet", "Cleaned and published 15 open datasets, with documentation volunteers could follow."),
    ("Heading 1", "SKILLS"),
    ("Normal", "SQL, Python, dbt, BigQuery, Looker, experiment design, causal inference, team leadership"),
    ("Heading 1", "EDUCATION"),
    ("Normal", "M.P.P., Quantitative Policy Analysis · B.A., Economics"),
]

OLD_RESUME = [
    ("Title", "JORDAN RIVERA"),
    ("Normal", "Washington, DC · jordan.rivera@example.com"),
    ("Heading 1", "EXPERIENCE"),
    ("job", "Fairway Health Partners | Senior Manager, Data Science | 2017 - Present"),
    ("bullet", "Manage 4 data scientists building enrollment forecasts for Medicaid outreach."),
    ("bullet", "Ran randomized outreach tests across 3 states."),
    ("job", "Open Streets Institute | Data Analyst | 2013 - 2017"),
    ("bullet", "Built the transit-equity dashboard used in council budget hearings."),
    ("Heading 1", "SKILLS"),
    ("Normal", "SQL, Python, R, Tableau, experiment design"),
]

COVER_LETTER = [
    ("Normal", "Dear Hiring Team,"),
    ("Normal", ""),
    (
        "Normal",
        "I've spent the last twelve years making public programs measurable, and I'd like to bring that "
        "work to Civic Signal. Your focus on helping agencies see whether a service is working for the "
        "people it's meant to reach is the problem I care most about.",
    ),
    ("Normal", ""),
    (
        "Normal",
        "At Common Ledger I lead a team of seven that supports forty benefit programs. We built an outcomes "
        "framework that's now part of every client contract, and moved reporting from spreadsheets to a "
        "warehouse program staff use on their own. Before that, I ran randomized outreach tests that raised "
        "Medicaid renewals by eleven percent.",
    ),
    ("Normal", ""),
    (
        "Normal",
        "I'd bring the same habits here: agree on what success means before launch, keep the team small and "
        "senior, and make the numbers useful to the people doing the work. I'd welcome the chance to talk.",
    ),
    ("Normal", ""),
    ("Normal", "Sincerely,"),
    ("Normal", "Jordan Rivera"),
]

OLD_COVER_LETTER = [
    ("Normal", "Dear Selection Committee,"),
    ("Normal", ""),
    (
        "Normal",
        "I'm writing to apply for the Senior Manager, Data Science role. At Open Streets Institute I built "
        "dashboards that shaped two city budgets, and I'm ready to lead a team doing the same at scale.",
    ),
    ("Normal", ""),
    ("Normal", "Thank you for your consideration,"),
    ("Normal", "Jordan Rivera"),
]

CANDIDATE_PROFILE = """## Career arc
Jordan is an analytics leader with about 12 years in public-sector and nonprofit data, moving from \
analyst (Open Streets Institute) to data science manager (Fairway Health Partners) to Director of \
Analytics at Common Ledger, a gov-tech firm serving state and city benefit programs.

## Strengths
- **Measurement that gets used:** outcomes frameworks, success metrics agreed before launch, \
reporting program staff run themselves.
- **Rigor:** randomized outreach tests and causal inference, with a documented 11% lift in renewals.
- **Modern data stack:** dbt, BigQuery and Looker; replaced spreadsheet reporting with a warehouse.
- **Leadership:** has built and led teams of 4-7; prefers small, senior teams.
- **Translation:** comfortable with agency leadership, engineers and program staff.

## What Jordan is looking for
- Director/Head/VP of Data or Analytics, or a senior individual role with real scope.
- Civic tech, public health, democracy and other mission-driven organizations.
- DC metro or remote; salary floor around $165k.

## Watch-outs
- Less experience with large-scale ML engineering or ad-tech style optimization.
- The most recent documents are current; the archived resume is from before the Common Ledger role."""

PREFERENCE_PROFILE = """Jordan reliably marks **civic tech, public health and democracy** roles as interesting, \
especially ones that own measurement or evaluation end to end. Roles at a **Director or Head level** with a \
small team get the strongest response; pure individual-contributor ML roles and anything centered on \
**advertising or growth marketing** are consistently passed over. Applications that reached interviews were \
mostly at **mission-driven organizations with 50-500 people**. Remote and DC-area roles are preferred; \
on-site roles elsewhere are skipped even when the work fits."""

SETTINGS = {
    "prefilter_titles_include": ["Director of Analytics", "Head of Data", "VP of Data", "Analytics Lead"],
    "prefilter_sectors": ["civic tech", "public health", "democracy", "nonprofit"],
    "prefilter_locations": ["DC", "Remote"],
    "location_dealbreaker": True,
    "salary_floor": 165000,
    "salary_dealbreaker": False,
    "looking_for_notes": (
        "I want to lead a small data team at a mission-driven org where measurement shapes real decisions. "
        "Government services, health access and democracy are the areas I care most about. I'm not "
        "interested in ad tech or growth marketing, however good the pay."
    ),
    "onboarding_completed": True,
    "onboarding_step": "done",
    "daily_llm_cap": 20,
}

# ---------------------------------------------------------------------------------------------
# Sources: (key, kind, name, company_name, config, hours since last check, last error). Organizations are
# web pages on .example addresses: a made-up Greenhouse/Lever/Ashby slug links to the real site.

SOURCES = [
    (
        "civic_signal",
        "custom_board",
        "Civic Signal",
        "Civic Signal",
        {"url": "https://civicsignal.example/careers"},
        3,
        None,
    ),
    (
        "open_ballot",
        "custom_board",
        "Open Ballot Lab",
        "Open Ballot Lab",
        {"url": "https://openballotlab.example/jobs"},
        3,
        None,
    ),
    (
        "brightwater",
        "custom_board",
        "Brightwater Health Data",
        "Brightwater Health Data",
        {"url": "https://brightwaterhealth.example/careers"},
        3,
        None,
    ),
    (
        "common_good",
        "custom_board",
        "Common Good Analytics",
        "Common Good Analytics",
        {"url": "https://jobs.commongood.example/careers"},
        3,
        None,
    ),
    (
        "harbor",
        "custom_board",
        "Harbor Benefits",
        "Harbor Benefits",
        {"url": "https://harborbenefits.example/careers"},
        3,
        None,
    ),
    (
        "northstar",
        "custom_board",
        "Northstar Public",
        "Northstar Public",
        {"url": "https://northstarpublic.example/jobs"},
        3,
        None,
    ),
    (
        "lantern",
        "custom_board",
        "Lantern Learning",
        "Lantern Learning",
        {"url": "https://lanternlearning.example/jobs"},
        27,
        "Couldn't read https://lanternlearning.example/jobs: the page timed out (they may be updating it). "
        "It'll be tried again at the next daily check.",
    ),
    (
        "tidewater",
        "custom_board",
        "Tidewater Climate",
        "Tidewater Climate",
        {"url": "https://tidewaterclimate.example/careers"},
        3,
        None,
    ),
    (
        "mission_data",
        "custom_board",
        "Mission Data Jobs",
        None,
        {"url": "https://missiondatajobs.example/"},
        3,
        None,
    ),
    (
        "public_interest",
        "custom_board",
        "Public Interest Tech Board",
        None,
        {"url": "https://publicinteresttech.example/jobs"},
        3,
        None,
    ),
    (
        "dc_impact",
        "custom_board",
        "DC Impact Careers",
        None,
        {"url": "https://dcimpactcareers.example/"},
        3,
        None,
    ),
    (
        "data4good",
        "custom_board",
        "Data for Good Network",
        None,
        {"url": "https://data4good.example/opportunities"},
        3,
        None,
    ),
    (
        "adzuna",
        "aggregator_adzuna",
        "Adzuna: analytics director",
        None,
        {"what": "director of analytics", "where": "Washington DC"},
        3,
        None,
    ),
    ("remoteok", "aggregator_remoteok", "RemoteOK: data", None, {"tags": ["data"]}, 3, None),
]

# ---------------------------------------------------------------------------------------------
# Jobs. Compact specs; seed.py fills in descriptions, dates and scores.
#   key, source, company (None = the source's own), title, location, remote_type,
#   salary (min, max, period) or None, posted days ago (None = unknown),
#   interest, fit (None = not scored yet), summary, rationale (with ==i:== / ==f:== marks)

JOBS = [
    (
        "cs_dir",
        "civic_signal",
        None,
        "Director of Data & Analytics",
        "Washington, DC",
        "hybrid",
        (175000, 205000, "year"),
        2,
        "High",
        "High",
        "Lead Civic Signal's 6-person data team measuring how state services reach residents.",
        "This is close to exactly what you've described: ==i:leading a small data team== at a civic tech "
        "organization where ==i:measurement shapes program decisions==. Your outcomes framework at Common Ledger "
        "maps directly onto their ==f:agency outcomes reporting==, and the ==f:dbt and BigQuery stack== is the one "
        "you already run. Pay is above your floor.",
    ),
    (
        "ob_head",
        "open_ballot",
        None,
        "Head of Analytics",
        "Remote (US)",
        "remote",
        (160000, 185000, "year"),
        5,
        "High",
        "Med",
        "Build Open Ballot Lab's analytics function from a team of two, supporting voter-access research.",
        "==i:Democracy work== and a chance to ==i:build the function from the start== both match what you want. "
        "Fit is a notch lower because the role leans on ==f:survey methodology==, which isn't prominent in your "
        "background, though your ==f:experiment design== experience carries over.",
    ),
    (
        "bw_vp",
        "brightwater",
        None,
        "VP, Data Strategy",
        "Remote (US)",
        "remote",
        (195000, 230000, "year"),
        1,
        "High",
        "High",
        "Own data strategy for a public-health data nonprofit serving 30 state health departments.",
        "==i:Public health== at a mission-driven nonprofit with a ==i:leadership remit== is a strong match. "
        "Your Medicaid outreach work at Fairway is ==f:directly relevant==, and they ask for someone who has "
        "==f:led teams of 5+==, which you have.",
    ),
    (
        "cg_lead",
        "common_good",
        None,
        "Analytics Lead, Programs",
        "Washington, DC",
        "onsite",
        (150000, 170000, "year"),
        8,
        "Med",
        "High",
        "Hands-on analytics lead for a nonprofit consultancy's program evaluation practice.",
        "The ==f:program evaluation== work is squarely in your wheelhouse and you'd be ==f:immediately "
        "effective==. Interest is moderate: it's a ==i:hands-on lead role without a team==, a step back from "
        "where you are.",
    ),
    (
        "hb_dir",
        "harbor",
        None,
        "Director, Analytics & Insights",
        "Arlington, VA",
        "hybrid",
        (170000, 195000, "year"),
        3,
        "High",
        "High",
        "Lead analytics for a benefits-navigation startup helping families enroll in SNAP and WIC.",
        "==i:Benefits access== is one of your core areas, and this is a ==i:director role with a team of "
        "five==. Your Common Ledger work on ==f:state benefit programs== and your ==f:warehouse migration== "
        "line up with their first-year goals.",
    ),
    (
        "ns_sr",
        "northstar",
        None,
        "Senior Data Scientist, Civic Platforms",
        "Remote (US)",
        "remote",
        (140000, 160000, "year"),
        12,
        "Med",
        "Med",
        "Senior individual-contributor data science role on a civic platform team.",
        "The ==i:civic mission== fits, but this is an ==i:individual-contributor role==, below the scope you're "
        "after. Your ==f:causal inference== background is relevant, though they want more ==f:production ML== "
        "than you've done recently.",
    ),
    (
        "td_head",
        "tidewater",
        None,
        "Head of Impact Measurement",
        "Remote (US)",
        "remote",
        (165000, 190000, "year"),
        4,
        "Med",
        "High",
        "Define and run impact measurement for a climate adaptation nonprofit.",
        "You'd be a ==f:strong fit for the measurement work== itself, and the ==f:outcomes framework== you built "
        "is the kind of thing they describe. Interest is Med because ==i:climate== isn't among your stated "
        "sectors, though the role's shape is right.",
    ),
    (
        "md_1",
        "mission_data",
        "Fieldwork Health",
        "Director of Analytics",
        "Remote (US)",
        "remote",
        (180000, 200000, "year"),
        6,
        "High",
        "High",
        "Lead analytics at a community health worker platform.",
        "==i:Public health access== plus a ==i:director-level team role==. Your Medicaid ==f:outreach "
        "experiments== and ==f:team leadership== are exactly what the posting emphasizes.",
    ),
    (
        "md_2",
        "mission_data",
        "Clearpath Legal Aid",
        "Data Manager",
        "Baltimore, MD",
        "onsite",
        (95000, 115000, "year"),
        9,
        "Low",
        "Med",
        "Manage case data and reporting for a legal aid organization.",
        "Mission-aligned, but ==i:well below your level and pay floor==, and on-site in Baltimore. You could "
        "certainly ==f:do the work==; it just isn't the next step you're looking for.",
    ),
    (
        "md_3",
        "mission_data",
        "Ballot Forward",
        "Analytics Director",
        "Washington, DC",
        "hybrid",
        None,
        3,
        "High",
        "Med",
        "Run analytics for a voter-registration nonprofit ahead of the next cycle.",
        "==i:Democracy and voter access== in DC, with a ==i:director title==. Fit is Med: they want deep "
        "==f:campaign and voter-file experience==, which you'd be learning.",
    ),
    (
        "pi_1",
        "public_interest",
        "US Digital Response Partners",
        "Data Practice Lead",
        "Remote (US)",
        "remote",
        (155000, 175000, "year"),
        2,
        "High",
        "High",
        "Lead a data practice helping governments modernize benefits delivery.",
        "A ==i:public-interest tech== role focused on ==i:government services==, which is where you've said you "
        "want to be. Your ==f:state program experience== and ==f:client-facing work== are a direct match.",
    ),
    (
        "pi_2",
        "public_interest",
        "Statewide Digital Service",
        "Chief Data Officer",
        "Richmond, VA",
        "hybrid",
        (185000, 210000, "year"),
        7,
        "High",
        "Med",
        "Statewide CDO role setting data policy and running a central analytics team.",
        "==i:Government data leadership== at the highest level you've targeted. Fit is Med because a CDO also "
        "owns ==f:data governance and policy==, broader than the analytics you've led.",
    ),
    (
        "pi_3",
        "public_interest",
        "Civic Signal",
        "Analytics Engineer",
        "Remote (US)",
        "remote",
        (130000, 150000, "year"),
        15,
        "Low",
        "Med",
        "Individual analytics engineering role on dbt models.",
        "You know the ==f:dbt stack== well, but this is an ==i:individual-contributor engineering role== below "
        "your pay floor.",
    ),
    (
        "dc_1",
        "dc_impact",
        "Capital Area Food Network",
        "Director of Evaluation & Learning",
        "Washington, DC",
        "hybrid",
        (145000, 165000, "year"),
        4,
        "Med",
        "High",
        "Lead evaluation for a regional food security network.",
        "Your ==f:evaluation and experiment design== background fits well. Interest is Med: the pay sits ==i:at "
        "the bottom of your range== and the team is two people.",
    ),
    (
        "dc_2",
        "dc_impact",
        "Metro Youth Alliance",
        "Data Coordinator",
        "Washington, DC",
        "onsite",
        (70000, 82000, "year"),
        10,
        "Low",
        "Low",
        "Entry-level data coordination for a youth services nonprofit.",
        "An ==i:entry-level role== far below your experience and pay. The work is ==f:mostly data entry==, which "
        "doesn't use your strengths.",
    ),
    (
        "d4g_1",
        "data4good",
        "Open Health Commons",
        "Head of Data",
        "Remote (US)",
        "remote",
        (170000, 190000, "year"),
        3,
        "High",
        "High",
        "First head of data at an open-source public health data project.",
        "==i:Public health==, ==i:head of data==, fully remote: this hits most of your stated priorities. They "
        "need someone who can ==f:build a warehouse from scratch== and ==f:work with agencies==, both of which "
        "you've done.",
    ),
    (
        "d4g_2",
        "data4good",
        "Brightline Ads",
        "Director of Data Science, Growth",
        "New York, NY",
        "onsite",
        (230000, 270000, "year"),
        2,
        "Low",
        "Med",
        "Lead growth data science for an advertising-technology company.",
        "Well paid, but ==i:ad tech and growth marketing== are what you've said you don't want, and it's "
        "==i:on-site in New York==. You'd bring ==f:experimentation skills==, though not the ad-auction depth "
        "they ask for.",
    ),
    (
        "d4g_3",
        "data4good",
        "Riverbend Schools Fund",
        "Senior Manager, Research & Data",
        "Remote (US)",
        "remote",
        (125000, 140000, "year"),
        20,
        "Med",
        "Med",
        "Research and data role at an education philanthropy.",
        "Mission-driven and remote, but ==i:education== isn't one of your focus areas and pay is ==i:below your "
        "floor==. Your ==f:research design== experience applies.",
    ),
    (
        "az_1",
        "adzuna",
        "Granite Federal Solutions",
        "Director, Analytics (Federal Health)",
        "Reston, VA",
        "hybrid",
        (175000, 200000, "year"),
        6,
        "Med",
        "High",
        "Lead analytics for a federal health contract at a government services firm.",
        "Your ==f:public health data== and ==f:agency-facing work== are what this contract needs. Interest is Med: "
        "it's a ==i:contractor role== rather than a mission-driven organization.",
    ),
    (
        "az_2",
        "adzuna",
        "Keystone Insurance Group",
        "Analytics Manager",
        "Washington, DC",
        "onsite",
        (120000, 140000, "year"),
        11,
        "Low",
        "Med",
        "Analytics manager at a commercial insurer.",
        "==i:Commercial insurance== is outside your sectors and the pay is below your floor. The ==f:analytics "
        "management== itself is familiar ground.",
    ),
    (
        "az_3",
        "adzuna",
        "Parkside Health System",
        "Director of Data Analytics",
        "Bethesda, MD",
        "hybrid",
        (165000, 185000, "year"),
        5,
        "Med",
        "High",
        "Lead analytics for a regional hospital system.",
        "==f:Health data== and a ==f:director-level team== fit well. Interest is Med: a hospital system is "
        "==i:health but not public health== in the sense you've described.",
    ),
    (
        "ro_1",
        "remoteok",
        "Openwell",
        "Head of Analytics",
        "Remote",
        "remote",
        (160000, 190000, "year"),
        3,
        "Med",
        "Med",
        "Head of analytics at a remote-first mental health startup.",
        "==i:Mental health access== is adjacent to your interests and it's a ==i:head-of role==. Fit is Med: "
        "they want ==f:product analytics for a consumer app==, which is newer ground for you.",
    ),
    (
        "ro_2",
        "remoteok",
        "Pixelmint",
        "Senior Data Analyst",
        "Remote",
        "remote",
        (110000, 130000, "year"),
        8,
        "Low",
        "Med",
        "Senior analyst at a mobile games studio.",
        "==i:Gaming== is far from your sectors and this is a ==i:senior analyst role==. You have the ==f:SQL and "
        "dashboarding skills== it asks for.",
    ),
    (
        "cs_2",
        "civic_signal",
        None,
        "Product Analyst",
        "Washington, DC",
        "hybrid",
        (105000, 125000, "year"),
        14,
        "Low",
        "Med",
        "Product analyst on Civic Signal's resident-facing tools.",
        "Right organization, ==i:wrong level==: this is an analyst role well under your experience. You'd "
        "==f:excel at the work== but it isn't the move you're after.",
    ),
    (
        "bw_2",
        "brightwater",
        None,
        "Director, Data Partnerships",
        "Remote (US)",
        "remote",
        (165000, 185000, "year"),
        6,
        "Med",
        "Med",
        "Manage data-sharing agreements with state health departments.",
        "==i:Public health== and remote, but the role is more ==i:partnerships than analytics==. Your "
        "==f:agency relationships== help; you'd have less hands-on data work than you want.",
    ),
    (
        "hb_2",
        "harbor",
        None,
        "Senior Analytics Engineer",
        "Arlington, VA",
        "hybrid",
        (145000, 165000, "year"),
        9,
        "Med",
        "High",
        "Senior analytics engineer building Harbor's dbt models.",
        "You'd be a ==f:strong technical fit== given your ==f:dbt and BigQuery== work. Interest is Med: it's an "
        "==i:individual-contributor role== at an organization you otherwise like.",
    ),
    (
        "ob_2",
        "open_ballot",
        None,
        "Research Director, Voting Access",
        "Remote (US)",
        "remote",
        (150000, 170000, "year"),
        18,
        "Med",
        "Low",
        "Lead qualitative and survey research on barriers to voting.",
        "==i:Voting access== is a cause you care about. Fit is Low: the role centers on ==f:qualitative "
        "research and survey fieldwork==, not analytics.",
    ),
    (
        "td_2",
        "tidewater",
        None,
        "Data Analyst, Climate Resilience",
        "Seattle, WA",
        "onsite",
        (90000, 105000, "year"),
        13,
        "Low",
        "Med",
        "Analyst on a climate resilience mapping project.",
        "==i:On-site in Seattle==, below your level and pay. The ==f:GIS and mapping== focus is also outside "
        "your core experience.",
    ),
    (
        "ns_2",
        "northstar",
        None,
        "Director of Product Analytics",
        "Austin, TX | Denver, CO",
        "onsite",
        (170000, 195000, "year"),
        4,
        "Med",
        "High",
        "Director of product analytics, on-site in Austin or Denver.",
        "A ==f:director-level analytics role== you'd do well in, at a civic platform you'd like. It's ==i:on-site "
        "in Austin or Denver==, which rules it out for you.",
    ),
    (
        "cg_2",
        "common_good",
        None,
        "Senior Evaluation Associate",
        "Washington, DC",
        "hybrid",
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "md_4",
        "mission_data",
        "Greenway Transit Alliance",
        "Head of Data & Research",
        "Washington, DC",
        "hybrid",
        (160000, 180000, "year"),
        1,
        None,
        None,
        None,
        None,
    ),
    (
        "pi_4",
        "public_interest",
        "Code for Counties",
        "Director of Data Services",
        "Remote (US)",
        "remote",
        (155000, 175000, "year"),
        0,
        None,
        None,
        None,
        None,
    ),
    (
        "dc_3",
        "dc_impact",
        "Housing Forward DC",
        "Data & Evaluation Manager",
        "Washington, DC",
        "hybrid",
        (110000, 125000, "year"),
        2,
        None,
        None,
        None,
        None,
    ),
    (
        "az_4",
        "adzuna",
        "Sterling Analytics Group",
        "Analytics Director",
        "Chantilly, VA",
        "onsite",
        (60, 75, "hour"),
        7,
        None,
        None,
        None,
        None,
    ),
    ("ro_3", "remoteok", "Lumen Civic", "Data Lead", "Remote", "remote", None, 2, None, None, None, None),
    # Set aside by the title screen (SET_ASIDE below): clear mismatches, never scored.
    ("ts_1", "mission_data", "Clearwater Legal Group", "Senior Litigation Counsel", "Washington, DC", "onsite", None, 1, None, None, None, None),
    ("ts_2", "dc_impact", "Beacon Health Services", "Registered Nurse, Night Shift", "Washington, DC", "onsite", None, 2, None, None, None, None),
    ("ts_3", "public_interest", "Northwind SaaS", "Enterprise Account Executive", "Remote (US)", "remote", None, 3, None, None, None, None),
    ("ts_4", "data4good", "Open Health Commons", "Summer Intern, Communications", "Remote (US)", "remote", None, 4, None, None, None, None),
]

# Set aside by the title screen as clear mismatches (never scored; hidden on All Jobs unless asked).
SET_ASIDE = {"ts_1", "ts_2", "ts_3", "ts_4"}

# Today's New & Noteworthy and yesterday's (job keys, best first is worked out by the app).
# A job on My Jobs never shows here (the app hides it), so these are mostly untracked ones. md_2
# (Baltimore) is hidden by the location dealbreaker, which shows that working.
DIGEST_TODAY = ["bw_vp", "md_1", "d4g_1", "td_head", "ro_1", "az_3", "hb_2", "d4g_3", "md_2"]
DIGEST_YESTERDAY = ["md_3", "ns_sr", "cs_2", "az_2", "ns_2", "pi_3"]

# My Jobs: job key, status, days since applied (None = not yet), notes, status history (status, days ago)
APPLICATIONS = [
    (
        "cs_dir",
        "interviewing",
        12,
        "Recruiter screen went well. Panel with the COO and two program leads next Tuesday.",
        [("saved", 15), ("waiting", 12), ("interviewing", 4)],
    ),
    (
        "hb_dir",
        "interviewing",
        9,
        "Take-home: outline a 90-day measurement plan for SNAP enrollment.",
        [("saved", 10), ("waiting", 9), ("interviewing", 2)],
    ),
    (
        "ob_head",
        "waiting",
        6,
        "Applied through their site; referral from a former colleague.",
        [("saved", 7), ("waiting", 6)],
    ),
    ("pi_1", "waiting", 3, None, [("saved", 3), ("waiting", 3)]),
    (
        "bw_2",
        "saved",
        None,
        "Tailor the resume toward public health partnerships before applying.",
        [("saved", 1)],
    ),
    ("ob_2", "saved", None, None, [("saved", 0)]),
    (
        "cg_lead",
        "rejected_no_interview",
        20,
        "Form rejection after two weeks.",
        [("saved", 22), ("waiting", 20), ("rejected_no_interview", 8)],
    ),
    (
        "pi_2",
        "rejected_interviewed",
        30,
        "Made the final round; they went with an internal candidate. Asked to stay in touch.",
        [("saved", 33), ("waiting", 30), ("interviewing", 21), ("rejected_interviewed", 6)],
    ),
    (
        "az_1",
        "withdrawn",
        18,
        "Withdrew: the role turned out to be mostly contract management.",
        [("saved", 19), ("waiting", 18), ("withdrawn", 11)],
    ),
    (
        "dc_1",
        "offer",
        26,
        "Offer: $162k. Asking for $172k and a start date after the holidays.",
        [("saved", 28), ("waiting", 26), ("interviewing", 17), ("offer", 2)],
    ),
]

# Feedback marks on jobs not in My Jobs (shapes the preference profile and the scoring experiment).
FEEDBACK = [
    ("d4g_2", "not_interested"),
    ("ro_2", "not_interested"),
    ("td_2", "not_interested"),
    ("ns_sr", "maybe"),
    ("td_head", "maybe"),
]

# Tailored resumes: job key -> edit plan against RESUME (exact anchors).
TAILORED_RESUMES = {
    "cs_dir": [
        (
            "replace",
            "Partner with product and policy leads to define success metrics before a program launches.",
            "Partner with agency, product and policy leads to agree on outcome metrics before a service launches.",
        ),
        (
            "insert_after",
            "Lead a team of 7 analysts and data engineers supporting 40 state and city benefit programs.",
            "Report program outcomes to state agency leadership quarterly, with metrics residents can see.",
        ),
    ],
    "hb_dir": [
        (
            "replace",
            "Lead a team of 7 analysts and data engineers supporting 40 state and city benefit programs.",
            "Lead a team of 7 analysts and data engineers supporting 40 benefit programs, including SNAP and WIC.",
        ),
        ("delete", "Built the transit-equity dashboard cited in two city council budget hearings.", None),
    ],
}
TAILORED_COVER_LETTERS = {
    "cs_dir": [
        "Dear Civic Signal team,",
        "I've spent twelve years making public programs measurable, and your Director of Data & Analytics role "
        "is the work I most want to be doing: helping agencies see whether services reach the residents "
        "they're meant for.",
        "At Common Ledger I lead seven analysts and engineers supporting forty benefit programs. We built an "
        "outcomes framework now written into every client contract and moved reporting onto a dbt and "
        "BigQuery warehouse that program staff run themselves, the same stack your team uses.",
        "I'd bring the habit of agreeing on outcomes before launch and keeping the numbers useful to the "
        "people doing the work. I'd welcome the chance to talk.",
        "Sincerely,",
        "Jordan Rivera",
    ],
}

# ---------------------------------------------------------------------------------------------
# Jobby: (job key or None, title for general chats, [(role, text), ...], days ago)

JOBBY = [
    (
        "cs_dir",
        None,
        1,
        [
            ("user", "Run a mock interview with me for this role."),
            (
                "assistant",
                "Let's do it. I'll play the hiring panel at Civic Signal: their COO and a program lead. One question "
                "at a time, and I'll give you feedback after each answer.\n\n**Question 1:** Civic Signal helps "
                'agencies see whether a service is working. Tell us about a time you defined what "working" meant '
                "for a program, and how you got everyone to agree on it.",
            ),
            (
                "user",
                "At Common Ledger we kept getting asked for reports after launch with no agreed definition of success. "
                "I set up a kickoff where the agency, our product lead and I picked three outcome metrics and a "
                "baseline before any build started. It's now in every contract, and reporting time dropped about 60%.",
            ),
            (
                "assistant",
                "That's a strong answer. You named the problem, what you did, and a result.\n\n**What worked:**\n"
                "- A concrete mechanism (the kickoff) rather than a general claim.\n"
                '- The 60% figure lands, and "now in every contract" shows it stuck.\n\n**To make it stronger:**\n'
                '- Name one metric you chose, so the panel can picture it. For example: "share of applicants who '
                'finish enrollment within 30 days."\n- Mention a disagreement and how you settled it. Panels '
                "listen for how you handle pushback from agencies.\n\n**Question 2:** Your team would be six people. "
                "How do you decide what *not* to work on?",
            ),
        ],
    ),
    (
        "hb_dir",
        None,
        2,
        [
            ("user", "What questions are they likely to ask me?"),
            (
                "assistant",
                "For Harbor Benefits' Director of Analytics & Insights role, expect questions in four areas:\n\n"
                "## The work itself\n- How would you measure whether a family who starts a SNAP application "
                "finishes it?\n- What would your first 90 days look like? (They've set a take-home on exactly this.)\n\n"
                "## Benefits programs\n- What makes enrollment data from different states hard to compare?\n- How do "
                "you handle data-sharing limits with state agencies?\n\n## Leadership\n- Tell us about growing a "
                "team from 3 to 7. What changed in how you managed?\n- How do you balance requests from product, "
                "operations and leadership?\n\n## Startup fit\n- You've worked at mid-sized organizations. What would "
                "change for you at a 60-person startup?\n\nYour strongest material is the warehouse migration and the "
                "outcomes framework. Want me to draft a 90-day plan outline you can build the take-home from?",
            ),
        ],
    ),
    (
        None,
        "Finding civic tech organizations in the DC area",
        4,
        [
            ("user", "Help me find organizations I should be watching."),
            (
                "assistant",
                "Based on your profile and the jobs you've rated highly, here's where I'd look beyond your current "
                "sources:\n\n## Government services\n- **Digital service teams** in state and county governments "
                'often hire data leads without posting on big boards. Look for "digital service" or '
                '"innovation office" pages.\n- **Benefits-navigation startups** like Harbor Benefits: you\'ve '
                "interviewed well with this kind of organization.\n\n## Public health\n- **Public health data "
                "nonprofits** that serve health departments (Brightwater and Open Health Commons both scored High "
                "for you).\n\n## Democracy\n- **Voter-access research groups** like Open Ballot Lab. Your interest "
                "is high here, but fit has been Med, so look for roles that lean on analytics over survey work.\n\n"
                "Your best interview rate so far is at **mission-driven organizations with 50-500 people**. I'd add "
                "sources in that range first.",
            ),
            ("user", "Which of these should I prioritize this month?"),
            (
                "assistant",
                "I'd start with **public health data nonprofits**. They're hiring at the director level right now "
                "(three High/High matches this week), the pay meets your floor, and they're mostly remote.\n\n"
                "Second, **benefits-navigation startups**: you're already interviewing at one, so you have fresh "
                "stories to reuse.\n\nI'd hold off on democracy organizations until closer to the next election "
                "cycle, when their analytics roles tend to open up.",
            ),
        ],
    ),
    (
        None,
        "Clarifying goals: director role vs. senior IC",
        9,
        [
            ("user", "Help me clarify my goals and strategy."),
            (
                "assistant",
                "Let's start with the decision that shapes the rest. Your applications split two ways:\n\n- "
                "**Director or Head roles** with a team (Civic Signal, Harbor Benefits, Capital Area Food Network)\n"
                "- **Senior individual roles** with deep technical work (Northstar's Senior Data Scientist, Harbor's "
                "Analytics Engineer)\n\nYour interviews have all come from the first group. What draws you to the "
                "second?",
            ),
            ("user", "Honestly I miss doing the analysis myself, but I like setting direction more."),
            (
                "assistant",
                "That's a common and useful tension to name. A few ways to keep both:\n\n1. **Target small teams.** "
                "At a 3-6 person team, a director still does real analysis. That fits the roles you've rated "
                'highest.\n2. **Ask about it directly** in interviews: "What share of this role is hands-on?"\n'
                "3. **Skip pure IC roles** unless the scope is unusual. Your scoring shows you rarely rate them "
                "highly once you read them.\n\nSo a clear strategy: director or head roles, small teams, "
                "mission-driven organizations, DC or remote. That's sharp enough to say in one breath when someone "
                "asks what you're looking for.",
            ),
        ],
    ),
]

APPLICATION_INSIGHTS = """You're getting interviews where the work is measurement for a public program, and \
not where it's general analytics.

## What's working
- **Civic tech and benefits organizations** account for every interview so far (Civic Signal, Harbor \
Benefits, the Statewide Digital Service, Capital Area Food Network).
- **High-fit roles convert.** Of your High-fit applications, most reached an interview, and one became an offer.
- **Director titles at small teams** are where your story lands best.

## What isn't
- **Hands-on lead roles** (Common Good's Analytics Lead) didn't get a response. Your resume reads as a \
leader, so individual roles may screen you out as over-qualified.
- **Contractor roles** (Granite Federal) weren't a match for what you want, which you found out late.

## What I'd try next
- Prioritize **public health data nonprofits**: three High/High matches this week, and none applied to yet.
- Ask about **team size and hands-on time** in the first screen, before investing in a take-home."""

DESCRIPTION_TEMPLATE = """## About {company}
{company} is a mission-driven organization working on {area}. We're a team of about {size} people, \
and we care about building things that measurably help the people we serve.

## The role
As {title}, you'll {verb} our work on data and evidence. {summary}

## What you'll do
- Set the direction for analytics and measurement, working closely with program and product leads.
- Define the outcomes we track, and make sure they're trusted and used.
- {extra}
- Share findings with leadership, partners and funders in plain language.

## What we're looking for
- {years}+ years of experience in analytics or data science, ideally in the public or nonprofit sector.
- Strong SQL and experience with a modern data stack (dbt, a cloud warehouse, a BI tool).
- Experience designing evaluations or experiments.
- A habit of explaining numbers so non-specialists can act on them.

## Compensation and benefits
{pay}Health, dental and vision coverage, a retirement match, and generous paid leave.

*{company} is an equal opportunity employer.* This posting is fictional, written for the Job Search \
Assistant demo."""
