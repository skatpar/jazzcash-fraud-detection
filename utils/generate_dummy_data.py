"""
Dummy Data Generator for JazzCash Fraud Detection

Generates realistic dummy data for testing the fraud detection pipeline without database access.

Generates:
- 100K+ user accounts (MBAR table)
- 400K+ transactions (IAR table)
- ~640 fraud cases (~0.16% fraud rate)

Usage:
    python utils/generate_dummy_data.py --num-users 100000 --num-transactions 400000
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse
from pathlib import Path
import hashlib


class DummyDataGenerator:
    """Generate realistic dummy data for fraud detection"""

    def __init__(self, num_users=100000, num_transactions=400000, fraud_rate=0.0016):
        """
        Initialize generator

        Args:
            num_users: Number of user accounts to generate
            num_transactions: Number of transactions to generate
            fraud_rate: Fraud rate (default 0.16%)
        """
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.num_frauds = int(num_transactions * fraud_rate)

        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        print(f"Initializing DummyDataGenerator:")
        print(f"  Users: {num_users:,}")
        print(f"  Transactions: {num_transactions:,}")
        print(f"  Fraud cases: {self.num_frauds:,} ({fraud_rate:.4%})")

    def _generate_phone_number(self):
        """Generate random Pakistani phone number"""
        prefix = random.choice(['0300', '0301', '0302', '0303', '0321', '0333', '0345'])
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return prefix + number

    def _encrypt_phone(self, phone):
        """Simulate encryption of phone number"""
        return hashlib.md5(phone.encode()).hexdigest()[:22] + "=="

    def _generate_account_reference(self, index):
        """Generate account reference"""
        return hashlib.md5(f"AC{index:010d}".encode()).hexdigest()[:22] + "=="

    def generate_mbar_data(self):
        """Generate MBAR (account/customer) data"""
        print("\nGenerating MBAR (account) data...")

        cities = [
            'KARACHI', 'LAHORE', 'ISLAMABAD', 'RAWALPINDI', 'FAISALABAD',
            'MULTAN', 'HYDERABAD', 'GUJRANWALA', 'PESHAWAR', 'QUETTA',
            'SIALKOT', 'BAHAWALPUR', 'SARGODHA', 'SUKKUR', 'LARKANA'
        ]

        provinces = ['PUNJAB', 'SINDH', 'KPK', 'BALOCHISTAN', 'ISLAMABAD']

        regions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL']

        account_types = [
            'L0 - Unverified', 'L1 - CNIC Verified', 'L2 - Biometric',
            'L3 - Full KYC', 'Agent Account', 'Merchant Account'
        ]

        channels = ['USSD', 'Mobile App', 'Web', 'Agent', 'ATM', 'POS']

        trust_levels = ['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']

        account_statuses = ['ACTIVE', 'DORMANT', 'SUSPENDED', 'CLOSED']

        # Generate registration dates (mostly in past 2 years)
        base_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 5, 31)

        data = []

        for i in range(self.num_users):
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i+1:,} accounts...")

            # Registration date
            days_diff = (end_date - base_date).days
            reg_date = base_date + timedelta(days=random.randint(0, days_diff))

            # Birth year (18-70 years old)
            birth_year = random.randint(1955, 2007)

            # City and province
            city = random.choice(cities)
            if city in ['KARACHI', 'HYDERABAD', 'SUKKUR', 'LARKANA']:
                province = 'SINDH'
            elif city in ['LAHORE', 'FAISALABAD', 'MULTAN', 'GUJRANWALA', 'SIALKOT', 'BAHAWALPUR', 'SARGODHA']:
                province = 'PUNJAB'
            elif city == 'PESHAWAR':
                province = 'KPK'
            elif city == 'QUETTA':
                province = 'BALOCHISTAN'
            else:
                province = 'ISLAMABAD'

            # Account reference (encrypted)
            ac_reference = self._generate_account_reference(i)

            # Phone number (encrypted)
            phone = self._generate_phone_number()
            gmsisdn = self._encrypt_phone(phone)

            # Account type (weighted)
            ac_type = random.choices(
                account_types,
                weights=[0.15, 0.30, 0.25, 0.20, 0.05, 0.05],
                k=1
            )[0]

            # Extract level from account type
            if 'L0' in ac_type:
                ac_level = 'L0'
            elif 'L1' in ac_type:
                ac_level = 'L1'
            elif 'L2' in ac_type:
                ac_level = 'L2'
            elif 'L3' in ac_type:
                ac_level = 'L3'
            elif 'Agent' in ac_type:
                ac_level = 'AGENT'
            else:
                ac_level = 'MERCHANT'

            # Status (mostly active)
            status = random.choices(
                account_statuses,
                weights=[0.80, 0.10, 0.05, 0.05],
                k=1
            )[0]

            # Trust level
            trust = random.choices(
                trust_levels,
                weights=[0.30, 0.45, 0.20, 0.05],
                k=1
            )[0]

            # Dormant/reactive dates (only for some)
            dormant_date = None
            re_active_date = None
            if status == 'DORMANT' or random.random() < 0.1:
                dormant_date = reg_date + timedelta(days=random.randint(180, 600))
                if random.random() < 0.5:
                    re_active_date = dormant_date + timedelta(days=random.randint(30, 180))

            # Last modified
            last_modified = reg_date + timedelta(days=random.randint(1, (end_date - reg_date).days))

            account = {
                'a_c_reference': ac_reference,
                'gmsisdn': gmsisdn,
                'registered_date_time': reg_date,
                'last_modified_date_time': last_modified,
                'year_of_birth': birth_year,
                'year_mdob': birth_year,  # Month/day of birth year component
                'place_of_birth': city,
                'city': city,
                'region': random.choice(regions),
                'prov': province,
                'account_type_name': ac_type,
                'a_c_level': ac_level,
                'a_c_status': status,
                'registered_channel': random.choice(channels),
                'trust_level': trust,
                'mpin_status': random.choice(['ACTIVE', 'INACTIVE', 'LOCKED']),
                'filer': random.choice(['Y', 'N']),
                'agent_group': f"AG{random.randint(1, 50):03d}" if 'Agent' in ac_type else None,
                'limit_group': f"LG{random.randint(1, 10):02d}",
                'charge_profile': f"CP{random.randint(1, 5):02d}",
                'credit_dl_ml_yl': random.randint(10000, 1000000),
                'debit_dl_ml_yl': random.randint(10000, 500000),
                'dormant_date': dormant_date,
                're_active_date': re_active_date
            }

            data.append(account)

        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} accounts")
        return df

    def generate_iar_data(self, df_mbar):
        """Generate IAR (transaction) data"""
        print("\nGenerating IAR (transaction) data...")

        # Get active accounts
        active_accounts = df_mbar[df_mbar['a_c_status'] == 'ACTIVE']['a_c_reference'].tolist()

        channels = ['NEW_JC_APP', 'PAYMENT GATEWAY', 'THIRD_PARTY_WEB', 'USSD', 'ATM']

        transaction_types = [
            'Transfer(C2C)', 'Transfer(C2B)', 'Online Payment',
            'IBFT Outgoing Customer', 'Bill Payment', 'Get Loan',
            'Cash Withdrawal', 'Mobile Topup'
        ]

        statuses = ['Completed', 'Failed', 'Pending']

        utility_companies = [
            'PTCL', 'K-Electric', 'SNGPL', 'SSGC', 'WAPDA',
            'JAZZ', 'TELENOR', 'UFONE', 'ZONG', None
        ]

        # Date range
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 7, 31)
        date_range_days = (end_date - start_date).days

        data = []

        for i in range(self.num_transactions):
            if (i + 1) % 50000 == 0:
                print(f"  Generated {i+1:,} transactions...")

            # Transaction date and time
            tx_date = start_date + timedelta(
                days=random.randint(0, date_range_days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )

            # Sender and receiver accounts
            ac_from = random.choice(active_accounts)
            ac_to = random.choice(active_accounts)

            # Ensure sender != receiver
            while ac_to == ac_from:
                ac_to = random.choice(active_accounts)

            # Customer MSISDN (sender's phone)
            sender_info = df_mbar[df_mbar['a_c_reference'] == ac_from].iloc[0]
            customer_msisdn = sender_info['gmsisdn'] if random.random() > 0.01 else None

            # Channel and type
            channel = random.choices(
                channels,
                weights=[0.50, 0.25, 0.10, 0.10, 0.05],
                k=1
            )[0]

            tx_type = random.choices(
                transaction_types,
                weights=[0.30, 0.20, 0.20, 0.10, 0.08, 0.05, 0.04, 0.03],
                k=1
            )[0]

            # Amount (log-normal distribution)
            amount = np.random.lognormal(mean=7.0, sigma=1.5)
            amount = max(100, min(amount, 100000))  # Cap between 100 and 100K
            amount = round(amount, 2)

            # Round amounts for some transactions
            if random.random() < 0.3:
                amount = round(amount / 100) * 100

            # Balances
            start_balance = np.random.lognormal(mean=8.5, sigma=1.8)
            start_balance = max(amount, min(start_balance, 500000))
            start_balance = round(start_balance, 2)

            # Fees
            if amount < 1000:
                fee = 0
                fed = 0
            else:
                fee = round(amount * random.uniform(0.001, 0.015), 2)
                fed = round(fee * 0.15, 2)

            end_balance = start_balance - amount - fee - fed

            # Status (mostly completed)
            status = random.choices(
                statuses,
                weights=[0.92, 0.06, 0.02],
                k=1
            )[0]

            # Utility company (only for bill payments)
            utility = None
            if 'Bill' in tx_type or 'Topup' in tx_type:
                utility = random.choice(utility_companies)

            # Transaction ID
            trans_id = f"{int(tx_date.timestamp())}{random.randint(100000, 999999)}"

            transaction = {
                'trans_id': trans_id,
                'data_date': tx_date.date(),
                'trans_initiate_time': tx_date,
                'customer_msisdn': customer_msisdn,
                'trx_channel': channel,
                'trx_type': tx_type,
                'trx_status': status,
                'ac_from': ac_from,
                'ac_to': ac_to,
                'start_balance': start_balance,
                'trx_amt': amount,
                'end_balance': end_balance,
                'fee': fee,
                'fed': fed,
                'utility_company': utility,
                'bill_ref_number': f"BR{random.randint(1000000, 9999999)}" if utility else None,
                'reason_type': f"Reason {random.randint(1, 50):03d}",
                'pur_of_remit': random.choice(['0101-Vendor Payments - Software', None]),
                'ec': None,
                'merchant_id': None
            }

            data.append(transaction)

        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} transactions")
        return df

    def generate_fraud_data(self, df_iar):
        """Generate fraud data"""
        print("\nGenerating fraud data...")

        # Select random transactions to mark as fraud
        fraud_indices = np.random.choice(
            len(df_iar),
            size=self.num_frauds,
            replace=False
        )

        fraud_transactions = df_iar.iloc[fraud_indices].copy()

        data = []

        for idx, row in fraud_transactions.iterrows():
            # Create fraud case
            complaint_date = row['trans_initiate_time'] + timedelta(
                hours=random.randint(1, 72)
            )
            resolved_date = complaint_date + timedelta(
                days=random.randint(1, 30)
            )

            fraud_case = {
                'complaint_num': f"FRD{random.randint(100000, 999999)}",
                'trans_id': row['trans_id'],
                'complaint_msisdn': row['customer_msisdn'],
                'fraud_msisdn': row['ac_to'],  # Receiver is fraudster
                'victim_msisdn': row['customer_msisdn'],
                'ac_from': row['ac_from'],
                'ac_to': row['ac_to'],
                'trx_amount': row['trx_amt'],
                'trx_channel': row['trx_channel'],
                'trx_type': row['trx_type'],
                'created_datetime': complaint_date,
                'resolved_datetime': resolved_date,
                'transaction_datetime': row['trans_initiate_time']
            }

            data.append(fraud_case)

        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} fraud cases")
        return df

    def save_data(self, df_mbar, df_iar, df_fraud, output_dir='data/dummy'):
        """Save generated data"""
        print(f"\nSaving data to {output_dir}...")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save as parquet
        df_mbar.to_parquet(f"{output_dir}/mbar_dummy.parquet", index=False)
        print(f"  ✓ Saved MBAR: {output_dir}/mbar_dummy.parquet")

        df_iar.to_parquet(f"{output_dir}/iar_dummy.parquet", index=False)
        print(f"  ✓ Saved IAR: {output_dir}/iar_dummy.parquet")

        df_fraud.to_parquet(f"{output_dir}/fraud_dummy.parquet", index=False)
        print(f"  ✓ Saved Fraud: {output_dir}/fraud_dummy.parquet")

        # Also save as CSV for inspection
        df_mbar.head(1000).to_csv(f"{output_dir}/mbar_sample.csv", index=False)
        df_iar.head(1000).to_csv(f"{output_dir}/iar_sample.csv", index=False)
        df_fraud.head(100).to_csv(f"{output_dir}/fraud_sample.csv", index=False)
        print(f"  ✓ Saved sample CSVs for inspection")

        # Save statistics
        stats = {
            'generation_date': datetime.now().isoformat(),
            'num_accounts': len(df_mbar),
            'num_transactions': len(df_iar),
            'num_fraud_cases': len(df_fraud),
            'fraud_rate': len(df_fraud) / len(df_iar),
            'date_range': f"{df_iar['data_date'].min()} to {df_iar['data_date'].max()}",
            'total_transaction_volume': df_iar['trx_amt'].sum(),
            'avg_transaction_amount': df_iar['trx_amt'].mean(),
        }

        with open(f"{output_dir}/data_stats.txt", 'w') as f:
            f.write("Dummy Data Generation Statistics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"  ✓ Saved statistics: {output_dir}/data_stats.txt")
        print("\n" + "=" * 60)
        print("Data generation complete!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Accounts (MBAR): {len(df_mbar):,}")
        print(f"  Transactions (IAR): {len(df_iar):,}")
        print(f"  Fraud Cases: {len(df_fraud):,} ({len(df_fraud)/len(df_iar):.4%})")
        print(f"  Date Range: {stats['date_range']}")
        print(f"  Total Volume: ${stats['total_transaction_volume']:,.2f}")
        print(f"  Avg Amount: ${stats['avg_transaction_amount']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dummy data for JazzCash fraud detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default data (100K users, 400K transactions)
  python utils/generate_dummy_data.py

  # Generate smaller dataset for quick testing
  python utils/generate_dummy_data.py --num-users 10000 --num-transactions 50000

  # Generate larger dataset
  python utils/generate_dummy_data.py --num-users 200000 --num-transactions 1000000
        """
    )

    parser.add_argument(
        '--num-users',
        type=int,
        default=100000,
        help='Number of user accounts to generate (default: 100000)'
    )

    parser.add_argument(
        '--num-transactions',
        type=int,
        default=400000,
        help='Number of transactions to generate (default: 400000)'
    )

    parser.add_argument(
        '--fraud-rate',
        type=float,
        default=0.0016,
        help='Fraud rate as decimal (default: 0.0016 = 0.16%%)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/dummy',
        help='Output directory (default: data/dummy)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JazzCash Dummy Data Generator")
    print("=" * 60)

    # Generate data
    generator = DummyDataGenerator(
        num_users=args.num_users,
        num_transactions=args.num_transactions,
        fraud_rate=args.fraud_rate
    )

    df_mbar = generator.generate_mbar_data()
    df_iar = generator.generate_iar_data(df_mbar)
    df_fraud = generator.generate_fraud_data(df_iar)

    generator.save_data(df_mbar, df_iar, df_fraud, args.output_dir)


if __name__ == "__main__":
    main()
