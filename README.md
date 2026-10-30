# AutoGen Customer Support — Six-Department RAG + Human Escalation

Six-department AutoGen support prototype with a Streamlit UI, department-specific document retrieval, persistent human escalation tickets, SMTP notifications, and LangSmith tracing.

## Requirements

Python 3.11 or 3.12 recommended. An OpenAI API key is required for AI routing and responses. FAISS and sentence-transformers run department retrieval locally; the first model download requires internet.

## Install (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `OPENAI_API_KEY`, `SUPPORT_ADMIN_PASSWORD`, and optional LangSmith and SMTP settings in `.env`. Start:

```powershell
python -m streamlit run app.py
```

## Pages

- **Customer Chat:** AutoGen router chooses HR, Finance, Sales, IT, Digital or Admin.
- **Escalation Tickets:** password-gated demo dashboard for ticket assignment, status changes, human responses, and event history.
- **Knowledge Admin:** upload PDF, TXT or Markdown policy documents. Retrieval uses local Hugging Face embeddings + FAISS inner-product search. Answers are grounded in retrieved excerpts and include source labels.
- **Monitoring:** ticket metrics and last request details. Logs also appear in the terminal and LangSmith when enabled.

## Routing and escalation

- **Finance:** every Finance request escalates by default, including payments, salary discrepancies, refunds and invoices. This is deliberately conservative.
- **HR:** uses retrieved HR policy documents. If no sufficiently similar document is found, creates a ticket rather than inventing a policy.
- **All departments:** an explicit human request triggers escalation.
- **Other requests:** receive a specialist AI response.

## Email notifications

SMTP settings: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS`. Set department destination emails, e.g. `FINANCE_SUPPORT_EMAIL`, or fallback `ESCALATION_ADMIN_EMAIL`. Ticket creation sends a notification to the department; if the customer supplies an email, an acknowledgment is attempted. Marking a ticket RESOLVED with a response sends that response to the customer. Failed/missing email configuration does not block ticket creation. Emails contain no full query text by default.

## LangSmith

Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT`. `@traceable` instruments the request pipeline, routing, department responses, department retrieval and ticket operations. AutoGen internal LLM calls may require additional instrumentation for model-level traces. Do not enable full prompt tracing with sensitive customer data without consent, masking and retention controls.

## Limitations before production

**This is an extended prototype, not production ready.** The password gate is demo-only; replace with server-side OIDC/SSO and role-based permissions. Streamlit session history is not automatically restored after refresh even though messages are persisted in SQLite. department embeddings/index are rebuilt on each request (the model itself is cached). PDF text extraction does not OCR scanned pages. SMTP sends synchronously with no retry queue. There is no worker service, SLA scheduler, tenant isolation, attachment malware scanning, per-user ticket ownership, or formal PII redaction. LLM routing may be imperfect. No real payroll/banking integrations are implemented. Do not upload real sensitive records until security controls and trace redaction are implemented.

## Suggested production improvements

FastAPI backend, PostgreSQL, persistent FAISS/pgvector index, identity-aware department retrieval, human assignment queues, email retry worker, structured outputs for router, policy-based financial intent detection, secrets management, rate limits, audit controls, and automated tests.

## Six department knowledge bases

Upload documents for **HR, Finance, Sales, IT, Digital, and Admin** from the Knowledge Admin page. Files are stored in `knowledge/<department>/`. All six departments have independent FAISS retrieval. Finance queries still create a human escalation ticket by default; Finance knowledge is reserved for a future approved, read-only workflow. Other departments use retrieved knowledge or escalate if no relevant evidence is found. This prototype rebuilds FAISS indexes per request and does not enforce document-level ACLs.


## Gmail App Password SMTP notifications

Enable Google Account 2-Step Verification, then generate an App Password at https://myaccount.google.com/apppasswords (availability depends on account settings). Set in `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=true
SMTP_USER=yourcompany@gmail.com
SMTP_PASSWORD=your_16_character_gmail_app_password
SMTP_FROM=yourcompany@gmail.com
HR_SUPPORT_EMAIL=hr@example.com
FINANCE_SUPPORT_EMAIL=finance@example.com
SALES_SUPPORT_EMAIL=sales@example.com
IT_SUPPORT_EMAIL=it@example.com
DIGITAL_SUPPORT_EMAIL=digital@example.com
ADMIN_SUPPORT_EMAIL=admin@example.com
ESCALATION_ADMIN_EMAIL=admin@example.com
```

Use an app password, never your normal Gmail password. Do not commit `.env` to Git. SMTP notifications are attempted when tickets are created and customer resolution emails are attempted when a ticket is marked RESOLVED. Ticket creation remains successful if SMTP delivery fails; see `notification_status` in Escalation Tickets. Sensitive ticket content is excluded from department notification emails.

**Validation:** Syntax-compiled only; requires real credentials and browser tests. The admin password is a demo gate, not enterprise authorization. Replace with SSO/RBAC before real employee or financial data.
