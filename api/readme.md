Create Virtual Environment
```bash
python -m venv venv
```

Activate Virtual Environment
```bash
venv\Scripts\activate
```

Install Requirements:
```bash
pip install -r requirements.txt
```

Start App
```bash
python main.py
```

Deploy
```bash
scp -r api root@209.38.4.53:/root
```

Start (on server)
```
. venv/bin/activate
nohup python3 main.py
```

Stop
```
ps aux | grep "main.py"
kill PID
```