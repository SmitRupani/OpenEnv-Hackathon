import sys
sys.path.insert(0, "/home/pujan/OpenEnv-Hackathon")
try:
    from verify import main
    main()
except Exception as e:
    print("Error:", e)
