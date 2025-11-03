import time
import pandas as pd
import matplotlib.pyplot as plt
import requests

def plot_boxplot(data):
    plt.boxplot(data)
    
    plt.title(f'Latency for Test Messages')
    plt.ylabel('Latency')
    plt.xlabel('Test Message')

    plt.savefig('load_test_boxplot.png')

def do_request(session, url, message, timeout=10):
    """Perform a single POST request and measure latency in ms."""
    payload = {"message": message}
    start = time.perf_counter()
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        status = resp.status_code
        label = ""
        try:
            body = resp.json()
            label = str(body.get("label", "")) if isinstance(body, dict) else ""
        except Exception:
            # non-json or parse error
            label = ""
    except Exception as e:
        # network error or timeout
        status = 0
        label = ""
    end = time.perf_counter()
    latency_ms = (end - start) * 1000.0
    return {"message": message, "latency_ms": latency_ms, "status": status, "label": label}

if __name__ == "__main__":
    base_url = 'http://ece444-pra5-env.eba-thdmp6v4.us-east-1.elasticbeanstalk.com'
    # base_url = 'http://127.0.0.1:8000'
    endpoint = '/predict'
    full_url = base_url + endpoint

    test_messages = [
        'The world is ending!!!!',
        'The Blue Jays won the world series!!!!',
        'Mark Carney is officially the prime minister of Canada.',
        'McLaren is on track to win the Formula 1 championship this year.'
    ]

    results = []

    # make API requests
    sess = requests.Session()
    for message in test_messages:
        for i in range(100):
            try:
                result = do_request(sess, full_url, message)
                print('result: ', result)
                results.append(result)
            except:
                print('An error occured...')
    
    df = pd.DataFrame(results)
    df.index += 1
    df.to_csv('load_test_results.csv', index_label='Req ID')

    latency_data = [result['latency_ms'] for result in results]

    plot_boxplot([
        latency_data[:100],
        latency_data[100:200],
        latency_data[200:300],
        latency_data[300:]
    ])

    sess.close()
