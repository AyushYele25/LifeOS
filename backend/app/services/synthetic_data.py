"""
LIFEOS Synthetic Data Generator

Generates realistic, reproducible Indian household financial data with:
- Configurable users, time horizons, and seed.
- Salaries, household expenses, EMIs, insurance, subscriptions, utilities.
- Seasonal variations (Diwali festive shopping, summer AC electricity spikes).
- Injected anomalies with ground-truth labels for ML evaluation.
- Synthetic documents (Insurance, Loans, Warranties, Utility bills, Receipts).
- Direct database seeding or CSV/JSON exports.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Realistic Indian constants
INDIAN_FIRST_NAMES = [
    "Aarav", "Aditi", "Advik", "Ananya", "Ankit", "Anushka", "Arjun", "Ayush",
    "Bhavna", "Chirag", "Deepak", "Devika", "Divya", "Gaurav", "Harsh", "Isha",
    "Ishaan", "Kavya", "Karan", "Kunal", "Manish", "Meera", "Neha", "Nikhil",
    "Pooja", "Pranav", "Priya", "Rahul", "Rhea", "Rishi", "Rohan", "Sakshi",
    "Sameer", "Sanjay", "Shreya", "Siddharth", "Sneha", "Tanvi", "Varun", "Vikram"
]

INDIAN_LAST_NAMES = [
    "Agarwal", "Bansal", "Bhatia", "Chopra", "Deshmukh", "Gupta", "Iyer", "Jain",
    "Joshi", "Kapoor", "Kulkarni", "Kumar", "Mehta", "Mishra", "Nair", "Patel",
    "Rao", "Reddy", "Sharma", "Singh", "Verma", "Yadav"
]

CITIES = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]

BANKS = [
    ("HDFC Bank", "HDFC"),
    ("State Bank of India", "SBI"),
    ("ICICI Bank", "ICICI"),
    ("Axis Bank", "AXIS"),
    ("Kotak Mahindra Bank", "KOTAK")
]

MERCHANTS = {
    "Groceries": ["Blinkit", "Zepto", "Instamart", "BigBasket", "DMart", "Nature's Basket", "Local Kirana Store"],
    "Food & Dining": ["Swiggy", "Zomato", "McDonald's", "Starbucks", "Dominos Pizza", "Haldiram's", "Chai Point", "Third Wave Coffee"],
    "Shopping": ["Amazon India", "Flipkart", "Myntra", "Zara", "Uniqlo", "Croma", "Reliance Digital", "Ajio"],
    "Transportation": ["Uber India", "Ola Cabs", "Namma Metro", "Delhi Metro", "BluSmart", "Auto Rickshaw UPI"],
    "Fuel": ["Indian Oil", "Bharat Petroleum", "HPCL", "Shell Fuel Station"],
    "Utilities": ["Tata Power", "BESCOM", "Adani Electricity", "BSES Rajdhani", "Indraprastha Gas", "Mahanagar Gas"],
    "Mobile & Internet": ["Airtel Prepaid/Postpaid", "Jio Infocomm", "ACT Fibernet", "Tata Play Fiber"],
    "Subscriptions": ["Netflix India", "Spotify India", "Amazon Prime", "Disney+ Hotstar", "YouTube Premium", "Cult.fit"],
    "Health & Medical": ["Apollo Pharmacy", "1mg", "Medplus", "Fortis Hospital", "Max Healthcare", "Dr. Lal PathLabs"],
    "Entertainment": ["BookMyShow", "PVR Cinemas", "INOX", "Smaaash", "Steam Games"],
    "Maintenance": ["Urban Company", "Society Maintenance Charge", "Local Electrician UPI", "Water Tanker Delivery"]
}

SUBSCRIPTION_PLANS = [
    ("Netflix India", Decimal("649.00"), "monthly"),
    ("Spotify India", Decimal("119.00"), "monthly"),
    ("Amazon Prime", Decimal("1499.00"), "yearly"),
    ("Disney+ Hotstar", Decimal("299.00"), "quarterly"),
    ("YouTube Premium", Decimal("129.00"), "monthly"),
    ("Cult.fit", Decimal("2499.00"), "monthly"),
    ("Airtel Broadband", Decimal("799.00"), "monthly")
]


def generate_synthetic_dataset(
    num_users: int = 100,
    months: int = 12,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Generate realistic, multi-user synthetic financial records with ground-truth anomalies."""
    random.seed(seed)
    
    today = date.today()
    start_date = today - timedelta(days=months * 30)

    users_data = []
    accounts_data = []
    transactions_data = []
    assets_data = []
    loans_data = []
    insurance_data = []
    subscriptions_data = []
    documents_data = []
    events_data = []
    anomalies_ground_truth = []

    for u_idx in range(num_users):
        user_id = str(uuid.uuid4())
        first_name = random.choice(INDIAN_FIRST_NAMES)
        last_name = random.choice(INDIAN_LAST_NAMES)
        full_name = f"{first_name} {last_name}"
        email = f"{first_name.lower()}.{last_name.lower()}{u_idx+1}@lifeos.demo"
        phone = f"+91{random.randint(7000000000, 9999999999)}"
        city = random.choice(CITIES)

        monthly_salary = Decimal(str(random.choice([35000, 50000, 75000, 95000, 120000, 160000, 220000])))
        rent_amount = Decimal(str(int(monthly_salary * Decimal(random.uniform(0.20, 0.32)) // 500 * 500)))

        user_record = {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "phone": phone,
            "currency": "INR",
            "timezone": "Asia/Kolkata",
            "city": city,
            "monthly_salary": float(monthly_salary),
            "created_at": (start_date - timedelta(days=30)).isoformat(),
        }
        users_data.append(user_record)

        # 1. Accounts
        bank_name, bank_code = random.choice(BANKS)
        savings_id = str(uuid.uuid4())
        cc_id = str(uuid.uuid4())

        accounts_data.append({
            "id": savings_id,
            "user_id": user_id,
            "name": f"{bank_name} Salary Account",
            "account_type": "savings",
            "bank_name": bank_name,
            "account_number_masked": f"XXXX-XXXX-{random.randint(1000, 9999)}",
            "current_balance": float(monthly_salary * Decimal(random.uniform(1.2, 3.5))),
            "is_primary": True,
        })

        accounts_data.append({
            "id": cc_id,
            "user_id": user_id,
            "name": f"{bank_name} Millennia Credit Card",
            "account_type": "credit_card",
            "bank_name": bank_name,
            "account_number_masked": f"XXXX-XXXX-XXXX-{random.randint(1000, 9999)}",
            "current_balance": float(Decimal(random.randint(2000, 35000))),
            "is_primary": False,
        })

        # 2. Assets (Vehicle / Electronics)
        car_asset_id = str(uuid.uuid4())
        assets_data.append({
            "id": car_asset_id,
            "user_id": user_id,
            "name": "Hyundai Creta SX" if float(monthly_salary) > 80000 else "Tata Tiago XT",
            "asset_type": "vehicle",
            "brand": "Hyundai" if float(monthly_salary) > 80000 else "Tata Motors",
            "model": "2023",
            "purchase_price": 1450000.0 if float(monthly_salary) > 80000 else 680000.0,
            "current_value": 1200000.0 if float(monthly_salary) > 80000 else 550000.0,
            "purchase_date": (start_date - timedelta(days=200)).isoformat(),
            "warranty_expiry": (start_date + timedelta(days=600)).isoformat(),
        })

        # 3. Loans
        has_loan = random.random() < 0.40
        loan_emi = Decimal("0")
        if has_loan:
            loan_id = str(uuid.uuid4())
            loan_emi = Decimal(str(int(monthly_salary * Decimal(random.uniform(0.12, 0.22)) // 100 * 100)))
            loans_data.append({
                "id": loan_id,
                "user_id": user_id,
                "name": "Vehicle Loan",
                "loan_type": "vehicle",
                "lender": bank_name,
                "account_number_masked": f"LN-XXXX-{random.randint(1000, 9999)}",
                "principal_amount": float(loan_emi * Decimal("48")),
                "interest_rate": 8.75,
                "emi_amount": float(loan_emi),
                "emi_day": 10,
                "start_date": start_date.isoformat(),
                "end_date": (start_date + timedelta(days=1460)).isoformat(),
                "status": "active"
            })
            documents_data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "title": f"Loan Agreement — {bank_name} Vehicle Loan",
                "document_type": "loan",
                "provider": bank_name,
                "reference_number": f"LN-{random.randint(100000, 999999)}",
                "emi_amount": float(loan_emi),
                "interest_rate": 8.75,
                "tenure_months": 48,
                "due_day": 10,
                "extracted_fields": {
                    "lender": bank_name,
                    "loan_type": "Vehicle Loan",
                    "emi": float(loan_emi),
                    "tenure": 48,
                    "due_date": "10th of every month"
                }
            })

        # 4. Insurance Policy
        ins_id = str(uuid.uuid4())
        ins_premium = Decimal(str(random.choice([12500, 18000, 24000])))
        renewal_date = start_date + timedelta(days=random.randint(60, 330))
        insurance_data.append({
            "id": ins_id,
            "user_id": user_id,
            "name": "Comprehensive Health Guard",
            "insurance_type": "health",
            "provider": random.choice(["Star Health", "HDFC ERGO", "Care Health"]),
            "policy_number": f"POL-H-{random.randint(1000000, 9999999)}",
            "premium_amount": float(ins_premium),
            "premium_frequency": "yearly",
            "sum_insured": 1000000.0,
            "start_date": (renewal_date - timedelta(days=365)).isoformat(),
            "expiry_date": renewal_date.isoformat(),
            "next_premium_date": renewal_date.isoformat(),
            "is_active": True,
        })
        documents_data.append({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": "Health Insurance Policy Schedule",
            "document_type": "insurance",
            "provider": "HDFC ERGO",
            "reference_number": f"POL-H-{random.randint(1000000, 9999999)}",
            "premium_amount": float(ins_premium),
            "expiry_date": renewal_date.isoformat(),
            "extracted_fields": {
                "policy_holder": full_name,
                "sum_insured": 1000000.0,
                "annual_premium": float(ins_premium),
                "due_date": renewal_date.isoformat(),
            }
        })

        # 5. Active Subscriptions
        user_subs = random.sample(SUBSCRIPTION_PLANS, k=random.randint(2, 4))
        for sub_name, sub_amount, freq in user_subs:
            sub_id = str(uuid.uuid4())
            subscriptions_data.append({
                "id": sub_id,
                "user_id": user_id,
                "name": sub_name,
                "provider": sub_name.split()[0],
                "amount": float(sub_amount),
                "frequency": freq,
                "billing_day": random.randint(1, 28),
                "status": "active"
            })

        # 6. Timeline of Transactions (12 months)
        inject_utility_spike = (u_idx % 5 == 0)
        inject_salary_drop = (u_idx % 8 == 0)
        inject_large_purchase = (u_idx % 6 == 0)
        inject_duplicate_txn = (u_idx % 4 == 0)

        current_cursor = start_date
        month_count = 0

        while current_cursor <= today:
            # Monthly Salary on Day 1
            salary_to_credit = monthly_salary
            if inject_salary_drop and month_count == 8:
                salary_to_credit = Decimal(str(int(monthly_salary * Decimal("0.55"))))
                anomalies_ground_truth.append({
                    "user_id": user_id,
                    "date": current_cursor.isoformat(),
                    "type": "salary_drop",
                    "expected": float(monthly_salary),
                    "observed": float(salary_to_credit),
                    "evidence": f"Salary dropped from ₹{monthly_salary} to ₹{salary_to_credit} (-45% unexpected decrease)"
                })

            transactions_data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "account_id": savings_id,
                "date": current_cursor.isoformat(),
                "description": f"ACH CREDIT — {bank_name} CORPORATE SALARY",
                "amount": float(salary_to_credit),
                "transaction_type": "credit",
                "merchant": "Employer Payroll",
                "is_recurring": True,
                "is_anomaly": (inject_salary_drop and month_count == 8),
                "category": "Salary"
            })

            # Monthly Rent on Day 3
            transactions_data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "account_id": savings_id,
                "date": (current_cursor + timedelta(days=2)).isoformat(),
                "description": f"UPI/HOUSE RENT/{random.choice(INDIAN_FIRST_NAMES)}/LANDLORD",
                "amount": float(rent_amount),
                "transaction_type": "debit",
                "merchant": "Landlord Rent UPI",
                "is_recurring": True,
                "is_anomaly": False,
                "category": "Rent"
            })

            # Monthly Loan EMI (if any) on Day 10
            if has_loan:
                transactions_data.append({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "account_id": savings_id,
                    "date": (current_cursor + timedelta(days=9)).isoformat(),
                    "description": f"NACH DEBIT — {bank_name} VEHICLE LOAN EMI",
                    "amount": float(loan_emi),
                    "transaction_type": "debit",
                    "merchant": f"{bank_name} Retail Loans",
                    "is_recurring": True,
                    "is_anomaly": False,
                    "category": "EMI & Loans"
                })

            # Monthly Utilities
            base_elec = Decimal(str(random.randint(1200, 2200)))
            if current_cursor.month in [4, 5, 6]:
                base_elec = Decimal(str(int(base_elec * Decimal(random.uniform(1.4, 1.8)))))

            is_elec_spike = (inject_utility_spike and month_count == 6)
            if is_elec_spike:
                elec_amount = base_elec * Decimal("3.8")
                anomalies_ground_truth.append({
                    "user_id": user_id,
                    "date": (current_cursor + timedelta(days=14)).isoformat(),
                    "type": "utility_spike",
                    "expected": float(base_elec),
                    "observed": float(elec_amount),
                    "evidence": f"Electricity bill ₹{elec_amount:.2f} is 280% higher than historical average of ₹{base_elec:.2f}"
                })
            else:
                elec_amount = base_elec

            transactions_data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "account_id": savings_id,
                "date": (current_cursor + timedelta(days=14)).isoformat(),
                "description": f"BBPS ELECTRICITY BILL — {random.choice(MERCHANTS['Utilities'])}",
                "amount": float(elec_amount),
                "transaction_type": "debit",
                "merchant": "Electricity Board",
                "is_recurring": True,
                "is_anomaly": is_elec_spike,
                "category": "Electricity"
            })

            # Subscriptions
            for s_name, s_amt, freq in user_subs:
                should_bill = (freq == "monthly") or (freq == "quarterly" and month_count % 3 == 0) or (freq == "yearly" and month_count == 2)
                if should_bill:
                    transactions_data.append({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "account_id": cc_id,
                        "date": (current_cursor + timedelta(days=random.randint(5, 25))).isoformat(),
                        "description": f"POS/E-COM RECURRING — {s_name.upper()}",
                        "amount": float(s_amt),
                        "transaction_type": "debit",
                        "merchant": s_name,
                        "is_recurring": True,
                        "is_anomaly": False,
                        "category": "Subscriptions"
                    })

            # Daily Living Expenses
            num_txns = random.randint(14, 22)
            for _ in range(num_txns):
                txn_day = random.randint(1, 28)
                txn_date = current_cursor + timedelta(days=txn_day)
                if txn_date > today:
                    continue

                cat = random.choices(
                    ["Groceries", "Food & Dining", "Transportation", "Fuel", "Shopping", "Entertainment"],
                    weights=[0.30, 0.25, 0.18, 0.12, 0.10, 0.05],
                    k=1
                )[0]

                merchant = random.choice(MERCHANTS[cat])

                if cat == "Groceries":
                    amt = Decimal(str(random.randint(450, 3800)))
                elif cat == "Food & Dining":
                    amt = Decimal(str(random.randint(180, 1650)))
                elif cat == "Transportation":
                    amt = Decimal(str(random.randint(80, 950)))
                elif cat == "Fuel":
                    amt = Decimal(str(random.randint(500, 3500)))
                elif cat == "Shopping":
                    if txn_date.month in [10, 11]:
                        amt = Decimal(str(random.randint(2500, 18500)))
                    else:
                        amt = Decimal(str(random.randint(800, 6500)))
                else:
                    amt = Decimal(str(random.randint(300, 1800)))

                txn_id = str(uuid.uuid4())
                transactions_data.append({
                    "id": txn_id,
                    "user_id": user_id,
                    "account_id": random.choice([savings_id, cc_id]),
                    "date": txn_date.isoformat(),
                    "description": f"UPI/PAYMENT TO {merchant.upper()}/REF-{random.randint(100000, 999999)}",
                    "amount": float(amt),
                    "transaction_type": "debit",
                    "merchant": merchant,
                    "is_recurring": False,
                    "is_anomaly": False,
                    "category": cat
                })

                # Injected Anomaly: Duplicate Charge
                if inject_duplicate_txn and random.random() < 0.015:
                    dup_id = str(uuid.uuid4())
                    transactions_data.append({
                        "id": dup_id,
                        "user_id": user_id,
                        "account_id": cc_id,
                        "date": txn_date.isoformat(),
                        "description": f"POS/DUPLICATE CHARGE — {merchant.upper()}",
                        "amount": float(amt),
                        "transaction_type": "debit",
                        "merchant": merchant,
                        "is_recurring": False,
                        "is_anomaly": True,
                        "category": cat
                    })
                    anomalies_ground_truth.append({
                        "user_id": user_id,
                        "date": txn_date.isoformat(),
                        "type": "duplicate_transaction",
                        "expected": 0,
                        "observed": float(amt),
                        "evidence": f"Identical amount ₹{amt} charged twice at {merchant} on the same day"
                    })
                    inject_duplicate_txn = False

            # Injected Anomaly: Sudden large luxury purchase
            if inject_large_purchase and month_count == 5:
                large_amt = Decimal(str(random.randint(75000, 115000)))
                transactions_data.append({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "account_id": cc_id,
                    "date": (current_cursor + timedelta(days=18)).isoformat(),
                    "description": "POS/RETAIL — APPLE STORE BKC MUMBAI",
                    "amount": float(large_amt),
                    "transaction_type": "debit",
                    "merchant": "Apple Store BKC",
                    "is_recurring": False,
                    "is_anomaly": True,
                    "category": "Shopping"
                })
                anomalies_ground_truth.append({
                    "user_id": user_id,
                    "date": (current_cursor + timedelta(days=18)).isoformat(),
                    "type": "unexpected_large_purchase",
                    "expected": float(monthly_salary * Decimal("0.10")),
                    "observed": float(large_amt),
                    "evidence": f"Sudden large purchase of ₹{large_amt} at Apple Store is 750% higher than average shopping transaction"
                })
                inject_large_purchase = False

            current_cursor = (current_cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
            month_count += 1

        # 7. Upcoming Future Obligations
        for offset_days, ev_title, ev_type, ev_amt in [
            (5, "Apartment Maintenance Due", "bill_due", Decimal("3500.00")),
            (12, f"{bank_name} Credit Card Payment", "bill_due", Decimal(random.randint(8000, 24000))),
            (25, "Electricity Bill Due", "bill_due", Decimal("2150.00")),
            (45, "Health Insurance Annual Premium", "insurance_renewal", ins_premium),
            (75, "Vehicle Service & Inspection", "vehicle_service", Decimal("4500.00")),
        ]:
            events_data.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "event_type": ev_type,
                "title": ev_title,
                "description": f"Automated obligation tracking for {full_name}",
                "event_date": (today + timedelta(days=offset_days)).isoformat(),
                "amount": float(ev_amt),
                "status": "upcoming",
                "confidence": 0.95,
                "is_recurring": True,
            })

    dataset = {
        "metadata": {
            "num_users": num_users,
            "months": months,
            "seed": seed,
            "total_transactions": len(transactions_data),
            "total_accounts": len(accounts_data),
            "total_documents": len(documents_data),
            "total_events": len(events_data),
            "total_anomalies_injected": len(anomalies_ground_truth),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "users": users_data,
        "accounts": accounts_data,
        "transactions": transactions_data,
        "assets": assets_data,
        "loans": loans_data,
        "insurance_policies": insurance_data,
        "subscriptions": subscriptions_data,
        "documents": documents_data,
        "events": events_data,
        "anomalies_ground_truth": anomalies_ground_truth,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "synthetic_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        with open(os.path.join(output_dir, "synthetic_transactions.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "user_id", "account_id", "date", "description",
                "amount", "transaction_type", "merchant", "category", "is_recurring", "is_anomaly"
            ])
            writer.writeheader()
            for t in transactions_data:
                writer.writerow(t)

        with open(os.path.join(output_dir, "anomalies_ground_truth.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user_id", "date", "type", "expected", "observed", "evidence"
            ])
            writer.writeheader()
            for a in anomalies_ground_truth:
                writer.writerow(a)

        print(f"✅ Generated dataset saved to {output_dir}:")
        print(f"   • {len(users_data)} users")
        print(f"   • {len(transactions_data)} transactions")
        print(f"   • {len(documents_data)} document records")
        print(f"   • {len(anomalies_ground_truth)} labeled anomalies")

    return dataset


async def seed_database_from_synthetic(dataset: Dict[str, Any], session: Optional[Any] = None):
    """Insert generated synthetic dataset into the database."""
    from sqlalchemy import select
    from app.database.session import async_session
    from app.services.auth_service import hash_password
    from app.models import (
        User, Account, Transaction, Asset, Loan, Insurance,
        Subscription, Event, Document
    )
    from app.models.models import (
        TransactionType, AccountType, AssetType, LoanType, LoanStatus,
        InsuranceType, SubscriptionFrequency, SubscriptionStatus,
        EventType, EventStatus, DocumentType
    )

    print("\n🌱 Seeding database from synthetic dataset...")

    async def _do_seed(s):
        # Fetch existing emails to prevent duplicate key errors
        existing_res = await s.execute(select(User.email))
        existing_emails = set(existing_res.scalars().all())

        user_ids_to_seed = set()
        for u in dataset["users"]:
            if u["email"] in existing_emails:
                continue
            user = User(
                id=uuid.UUID(u["id"]),
                email=u["email"],
                hashed_password=hash_password("DemoPassword123!"),
                full_name=u["full_name"],
                phone=u["phone"],
                currency=u["currency"],
                timezone=u["timezone"],
            )
            s.add(user)
            user_ids_to_seed.add(u["id"])

        await s.flush()

        # Seed Accounts
        for a in dataset["accounts"]:
            if a["user_id"] in user_ids_to_seed:
                acc = Account(
                    id=uuid.UUID(a["id"]),
                    user_id=uuid.UUID(a["user_id"]),
                    name=a["name"],
                    account_type=AccountType(a["account_type"]),
                    bank_name=a["bank_name"],
                    account_number_masked=a["account_number_masked"],
                    current_balance=Decimal(str(round(a["current_balance"], 2))),
                    is_primary=a["is_primary"],
                )
                s.add(acc)

        # Seed Assets
        for asset in dataset["assets"]:
            if asset["user_id"] in user_ids_to_seed:
                ass = Asset(
                    id=uuid.UUID(asset["id"]),
                    user_id=uuid.UUID(asset["user_id"]),
                    name=asset["name"],
                    asset_type=AssetType(asset["asset_type"]),
                    brand=asset["brand"],
                    model=asset["model"],
                    purchase_price=Decimal(str(asset["purchase_price"])),
                    current_value=Decimal(str(asset["current_value"])),
                    purchase_date=date.fromisoformat(asset["purchase_date"]),
                    warranty_expiry=date.fromisoformat(asset["warranty_expiry"]),
                )
                s.add(ass)

        # Seed Loans
        for loan in dataset["loans"]:
            if loan["user_id"] in user_ids_to_seed:
                ln = Loan(
                    id=uuid.UUID(loan["id"]),
                    user_id=uuid.UUID(loan["user_id"]),
                    name=loan["name"],
                    loan_type=LoanType(loan["loan_type"]),
                    lender=loan["lender"],
                    account_number_masked=loan["account_number_masked"],
                    principal_amount=Decimal(str(loan["principal_amount"])),
                    interest_rate=loan["interest_rate"],
                    emi_amount=Decimal(str(loan["emi_amount"])),
                    emi_day=loan["emi_day"],
                    start_date=date.fromisoformat(loan["start_date"]),
                    end_date=date.fromisoformat(loan["end_date"]),
                    status=LoanStatus.ACTIVE,
                )
                s.add(ln)

        # Seed Insurance
        for ins in dataset["insurance_policies"]:
            if ins["user_id"] in user_ids_to_seed:
                insp = Insurance(
                    id=uuid.UUID(ins["id"]),
                    user_id=uuid.UUID(ins["user_id"]),
                    name=ins["name"],
                    insurance_type=InsuranceType(ins["insurance_type"]),
                    provider=ins["provider"],
                    policy_number=ins["policy_number"],
                    premium_amount=Decimal(str(ins["premium_amount"])),
                    premium_frequency=SubscriptionFrequency.YEARLY,
                    sum_insured=Decimal(str(ins["sum_insured"])),
                    start_date=date.fromisoformat(ins["start_date"]),
                    expiry_date=date.fromisoformat(ins["expiry_date"]),
                    next_premium_date=date.fromisoformat(ins["next_premium_date"]),
                    is_active=ins["is_active"],
                )
                s.add(insp)

        # Seed Subscriptions
        for sub in dataset["subscriptions"]:
            if sub["user_id"] in user_ids_to_seed:
                s_obj = Subscription(
                    id=uuid.UUID(sub["id"]),
                    user_id=uuid.UUID(sub["user_id"]),
                    name=sub["name"],
                    provider=sub["provider"],
                    amount=Decimal(str(sub["amount"])),
                    frequency=SubscriptionFrequency(sub["frequency"]),
                    next_billing_date=date.today() + timedelta(days=sub.get("billing_day", 15)),
                    category="Subscriptions",
                    status=SubscriptionStatus.ACTIVE,
                )
                s.add(s_obj)

        # Seed Events
        for ev in dataset["events"]:
            if ev["user_id"] in user_ids_to_seed:
                e = Event(
                    id=uuid.UUID(ev["id"]),
                    user_id=uuid.UUID(ev["user_id"]),
                    event_type=EventType(ev["event_type"]),
                    title=ev["title"],
                    description=ev["description"],
                    event_date=date.fromisoformat(ev["event_date"]),
                    amount=Decimal(str(ev["amount"])),
                    status=EventStatus(ev["status"]),
                    confidence=ev["confidence"],
                    is_recurring=ev["is_recurring"],
                )
                s.add(e)

        # Seed Documents
        for doc in dataset["documents"]:
            if doc["user_id"] in user_ids_to_seed:
                d = Document(
                    id=uuid.UUID(doc["id"]),
                    user_id=uuid.UUID(doc["user_id"]),
                    filename=f"{doc['title'].lower().replace(' ', '_')}.pdf",
                    original_filename=f"{doc['title']}.pdf",
                    file_path=f"uploads/{doc['id']}.pdf",
                    file_size=1024 * 45,
                    mime_type="application/pdf",
                    document_type=DocumentType(doc["document_type"]),
                    extracted_entities=doc["extracted_fields"],
                    is_processed=True,
                )
                s.add(d)

        # Seed Transactions in chunks
        batch_size = 500
        for i in range(0, len(dataset["transactions"]), batch_size):
            batch = dataset["transactions"][i:i + batch_size]
            for t in batch:
                if t["user_id"] in user_ids_to_seed:
                    txn = Transaction(
                        id=uuid.UUID(t["id"]),
                        user_id=uuid.UUID(t["user_id"]),
                        account_id=uuid.UUID(t["account_id"]),
                        date=date.fromisoformat(t["date"]),
                        description=t["description"],
                        amount=Decimal(str(t["amount"])),
                        transaction_type=TransactionType(t["transaction_type"]),
                        merchant=t["merchant"],
                        is_recurring=t["is_recurring"],
                        is_anomaly=t["is_anomaly"],
                        source="synthetic",
                    )
                    s.add(txn)
            await s.flush()

        await s.commit()
        print(f"🎉 Successfully imported synthetic data into database!")
        print(f"   • New Users Seeded: {len(user_ids_to_seed)}")

    if session is not None:
        await _do_seed(session)
    else:
        async with async_session() as s:
            await _do_seed(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIFEOS Synthetic Data Generator")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate (default: 100)")
    parser.add_argument("--months", type=int, default=12, help="Number of months of history (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory for generated files")
    parser.add_argument("--seed-db", action="store_true", help="Directly seed the generated data into the database")
    args = parser.parse_args()

    data = generate_synthetic_dataset(
        num_users=args.users,
        months=args.months,
        seed=args.seed,
        output_dir=args.output_dir
    )

    if args.seed_db:
        asyncio.run(seed_database_from_synthetic(data))
