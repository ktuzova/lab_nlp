"""
Скрипт инференса LLM Qwen2.5:0.5B через Ollama API.

Отправляет 10 запросов на локальный сервер Ollama,
собирает ответы и формирует отчёт в формате Markdown.
"""

import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
REPORT_FILE = "report.md"


QUERIES = [
    "What is the capital of France?",
    "Explain what machine learning is in 2-3 sentences.",
    "Write a short poem about the sea.",
    "What is the difference between Python and Java?",
    "Name 3 advantages of renewable energy.",
    "What is Docker and why is it used?",
    "Explain the concept of recursion in programming.",
    "What are the main causes of climate change?",
    "Translate to French: 'Good morning, how are you?'",
    "What is the Pythagorean theorem?",
]


def send_query(query: str) -> str:
    """Отправляет запрос к модели Qwen2.5:0.5B на сервере Ollama.

    Args:
        query: Текст запроса к LLM.

    Returns:
        Строка с ответом модели. В случае ошибки возвращает сообщение об ошибке.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": query,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ОШИБКА: Не удалось подключиться к серверу Ollama. Убедитесь, что Ollama запущена."
    except requests.exceptions.Timeout:
        return "ОШИБКА: Превышено время ожидания ответа."
    except requests.exceptions.RequestException as e:
        return f"ОШИБКА: {e}"


def run_inference(queries: list[str]) -> list[dict]:
    """Выполняет инференс для списка запросов.

    Args:
        queries: Список текстовых запросов к LLM.

    Returns:
        Список словарей с ключами 'query' и 'response'.
    """
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Отправка запроса: {query[:50]}...")
        answer = send_query(query)
        results.append({"query": query, "response": answer})
        print(f"  -> Ответ получен ({len(answer)} символов)")
    return results


def save_report(results: list[dict], filepath: str) -> None:
    """Сохраняет отчёт инференса в формате Markdown.

    Args:
        results: Список словарей с ключами 'query' и 'response'.
        filepath: Путь к файлу для сохранения отчёта.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Отчёт инференса Qwen2.5:0.5B\n\n")
        f.write(f"Модель: `{MODEL_NAME}`\n\n")
        f.write(f"Количество запросов: {len(results)}\n\n")
        f.write("| № | Запрос | Ответ LLM |\n")
        f.write("|---|--------|----------|\n")

        for i, item in enumerate(results, 1):
            query = item["query"].replace("|", "\\|").replace("\n", " ")
            response = item["response"].replace("|", "\\|").replace("\n", " ")
            f.write(f"| {i} | {query} | {response} |\n")

    print(f"\nОтчёт сохранён в {filepath}")


def main():
    """Главная функция: запускает инференс и сохраняет отчёт."""
    print(f"Запуск инференса модели {MODEL_NAME}...")
    print(f"Сервер Ollama: {OLLAMA_URL}\n")

    results = run_inference(QUERIES)
    save_report(results, REPORT_FILE)

    print("Готово!")


if __name__ == "__main__":
    main()
