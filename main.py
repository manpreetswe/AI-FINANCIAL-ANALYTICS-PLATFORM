# main.py
# AI Financial Analytics Platform Controller

import subprocess


while True:

    print("\n" + "=" * 70)
    print("        AI FINANCIAL ANALYTICS PLATFORM")
    print("=" * 70)

    print("1. Stock + SEC Downloader")
    print("2. Data Preprocessing")
    print("3. Database Manager")
    print("4. ML Forecasting Model")
    print("5. Investment Classifier")
    print("6. Dashboard")
    print("7. AI Chatbot")
    print("8. Run Full Streamlit App")
    print("9. Exit")

    print("=" * 70)

    choice = input("Enter Choice: ")

    try:

        # 1 Downloader
        if choice == "1":

            print("\nRunning Stock + SEC Downloader...\n")

            subprocess.run(
                "python data_collection/stock_downloader.py",
                shell=True
            )

            subprocess.run(
                "python data_collection/sec_downloader.py",
                shell=True
            )

        # 2 Preprocessing
        elif choice == "2":

            print("\nRunning Data Preprocessing...\n")

            subprocess.run(
                "python preprocessing/spark_preprocessor.py",
                shell=True
            )

        # 3 Database
        elif choice == "3":

            print("\nRunning Database Manager...\n")

            subprocess.run(
                "python sql_interface/database_manager.py",
                shell=True
            )

        # 4 ML Forecasting
        elif choice == "4":

            print("\nRunning ML Forecasting Model...\n")

            subprocess.run(
                "python ml_models/spark_gbt_forecaster.py",
                shell=True
            )

        # 5 Investment Classifier
        elif choice == "5":

            print("\nRunning Investment Classifier...\n")

            subprocess.run(
                "python ml_models/investment_classifier.py",
                shell=True
            )

        # 6 Dashboard
        elif choice == "6":

            print("\nRunning Dashboard...\n")

            subprocess.run(
                "python dashboard/dashboard_app.py",
                shell=True
            )

        # 7 AI Chatbot
        elif choice == "7":

            print("\nRunning AI Chatbot...\n")

            subprocess.run(
                "python chatbot/ai_prediction_chatbot.py",
                shell=True
            )

        # 8 Full Streamlit App
        elif choice == "8":

            print("\nLaunching Full Streamlit App...\n")

            subprocess.run(
                "streamlit run Home.py",
                shell=True
            )

        # 9 Exit
        elif choice == "9":

            print("\nExiting Platform...\n")
            break

        else:
            print("\nInvalid Choice")

    except Exception as e:
        print(f"\nError: {e}")
