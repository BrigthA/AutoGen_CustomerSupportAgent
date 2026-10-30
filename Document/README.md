# Synthetic RAG Policy PDF Pack

Six fictional example PDFs for testing the AutoGen customer support application. NOT legal or company policy.

Upload each PDF under the matching department in Knowledge Admin and ingest/rebuild its FAISS index. Do not upload confidential real employee records into this demo.

| PDF | Test query |
|---|---|
| HR_Employee_Policies.pdf | How many annual leave days do employees receive? |
| Finance_Billing_and_Payments.pdf | Can I get a refund for a duplicate charge? |
| Sales_Products_and_Quotations.pdf | Can I get a quotation for 150 seats? |
| IT_Helpdesk_and_Security.pdf | What should I do if my VPN will not connect? |
| Digital_Website_and_Marketing.pdf | How do I request a website content update? |
| Admin_Facilities_and_Access.pdf | What do I do if I lose my office access card? |

Expected: HR and other knowledge agents cite relevant sections; Finance routes to human escalation by default. Missing facts and sensitive actions should not be invented or executed.
