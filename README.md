
This code is developed and tested in `python 3.8.9`
It should work with Python version 3.6+

Firstly, let's create a virtual environment:
```zsh
python3 -m venv ~/.virtual_envs/data_viz && source ~/.virtual_envs/data_viz/bin/activate
```

Installing & upgrading packages:
```zsh
pip install pip -U && pip install -r requirements.txt
```

We can now run the WebApp via:
```
python -m streamlit run app.py
```

It will return and direct you to the browser (Chrome recommended):
```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.29:8501
```
In browser, address should be: `http://localhost:8501`

If you want to use another port, please run the command with desired port:
`python -m streamlit run app.py --server.port 4848`

You can also reach this website through your phone if your phone is connected to same network (WiFi) via:
`http://192.168.0.29:8501`