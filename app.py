import ast
import json
import operator
import re
import requests

from ddgs import DDGS
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates


# =========================================================
# SAKAI
# =========================================================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:1.7b"


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# CLEAN MODEL OUTPUT
# =========================================================

def clean_answer(text: str) -> str:
    if not text:
        return ""

    # Remove complete Qwen thinking blocks.
    text = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove an unfinished thinking block too.
    text = re.sub(
        r"<think>[\s\S]*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)

    # Remove common accidental model prefixes.
    text = re.sub(
        r"^\s*(assistant|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove common leaked reasoning prefixes.
    text = re.sub(
        r"(?is)^\s*(okay|ok|alright|sure|hmm|let me|let's|first|now|the user|i need to|i should|thinking|analysis)\b.*?(?=\b(?:output|answer|result)\s*[:=-])",
        "",
        text,
        count=1,
    )
    return text.strip()


# =========================================================
# VERY SIMPLE / FIXED ANSWERS
# =========================================================

def direct_answer(question: str):
    q = re.sub(r"\s+", " ", question.lower().strip())

    answers = {
        "hello": "Hello! 👋",
        "hi": "Hello! 👋",
        "hii": "Hello! 👋",
        "hey": "Hello! 👋",
        "who are you": "I am SakAI, your AI assistant. 🤖",
        "who r you": "I am SakAI, your AI assistant. 🤖",
        "what is the capital of india": "New Delhi",
        "capital of india": "New Delhi",
        "who was the first prime minister of india": "Jawaharlal Nehru",
        "first prime minister of india": "Jawaharlal Nehru",
        "national animal of india": "Bengal Tiger",
        "national bird of india": "Indian Peacock",
        "national flower of india": "Lotus",
    }

    return answers.get(q)


# =========================================================
# WEB SEARCH
# Only use it for information that can actually change.
# Stable questions should be answered directly by the model.
# =========================================================

def needs_web_search(question: str) -> bool:
    q = question.lower().strip()

    current_words = [
        "latest", "current", "today", "today's", "right now",
        "currently", "recent", "recently", "this year",
        "this month", "news", "live", "weather",
        "price today", "stock price", "score today",
        "result today", "now"
    ]

    return any(word in q for word in current_words)


def web_search(question: str) -> str:
    try:
        results = DDGS().text(question, max_results=4)

        if not results:
            return ""

        data = []

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            link = result.get("href", "")

            data.append(
                f"Title: {title}\n"
                f"Information: {body}\n"
                f"Source: {link}"
            )

        return "\n\n".join(data)

    except Exception as e:
        print("WEB SEARCH ERROR:", repr(e))
        return ""


# =========================================================
# SAFE MATH CALCULATOR
# =========================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError

    if isinstance(node, ast.BinOp):
        left = calculate_node(node.left)
        right = calculate_node(node.right)

        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError

        # Prevent absurd exponent calculations.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        value = calculate_node(node.operand)

        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError

        return operation(value)

    raise ValueError


def calculate_math(expression):
    try:
        expression = expression.replace("^", "**").strip()

        if len(expression) > 100:
            return None

        if not re.fullmatch(r"[0-9+\-*/%().\s^]+", expression):
            return None

        tree = ast.parse(expression, mode="eval")
        result = calculate_node(tree.body)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return result

    except Exception:
        return None


def get_math_expression(question: str):
    q = question.lower().strip()

    q = re.sub(
        r"\b(calculate|solve|evaluate|find|what is|answer)\b",
        "",
        q
    )

    q = q.replace("?", "").strip()

    if re.fullmatch(r"[0-9+\-*/%().\s^]+", q):
        return q

    return None


# =========================================================
# PROMPT
# =========================================================

def create_prompt(question: str, web_information: str = "") -> str:
    web_block = ""

    if web_information:
        web_block = f"""
CURRENT WEB INFORMATION:
{web_information}

Use this information only when it directly helps answer the user's question.
Do not copy source text repeatedly.
Do not mention that web search was used.
"""

    only_mode = bool(re.search(
        r"\b(only|just)\s+(output|answer|name|result|code)\b|"
        r"\bgive me only\b|\bonly give\b",
        question.lower()
    ))

    only_instruction = ""
    if only_mode:
        only_instruction = """
IMPORTANT OUTPUT MODE:
The user explicitly asked for ONLY the requested result.
Return ONLY the final requested result.
Do NOT explain.
Do NOT mention missing input unless absolutely necessary.
Do NOT show reasoning, analysis, thinking, or planning.
Do NOT use phrases such as "Okay, let me", "First", "Hmm", "The user", "I need to", or "Let's".
Do NOT repeat the question.
"""

    return f"""
You are SakAI, a helpful, accurate and concise AI assistant.
{only_instruction}

USER QUESTION:
{question}

{web_block}

STRICT RULES:

1. Answer ONLY the user's question.
2. Give exactly as much information as the question requires:
   - A short question -> a short answer.
   - A detailed question -> a detailed answer.
   - If the user asks for a specific number of words, follow that limit as closely as possible.
3. NEVER repeat the same answer, sentence, definition, or paragraph.
4. NEVER show internal reasoning, chain-of-thought, hidden thinking, <think> tags, or analysis.
5. Do not say "Thinking..." or describe your reasoning process.
6. Do not repeat the user's question.
7. Use normal spaces between every word.
8. Preserve paragraph breaks and use a blank line between separate paragraphs.
9. Use clear headings, numbered points, or bullet points when they make the answer easier to read.
10. Use correct punctuation.
11. Answer in the language/style requested by the user. If the user writes Hindi/Hinglish, reply in Hindi/Hinglish unless another language is requested.
12. For mathematics, calculate carefully and give the correct final result.
13. For programming questions, give correct code/output/explanation exactly as requested.
14. If programming output depends on missing input, provide a concise sample input and its corresponding output. If the user explicitly asks for ONLY output, return only the sample output.
15. For exam/college questions, keep the explanation appropriate for BCA/college level when relevant.
16. Do not invent facts. If you are uncertain, say so briefly.
17. Do not unnecessarily introduce yourself.
18. Return ONLY the final answer for the user.

Now answer the user question.
""".strip()


# =========================================================
# RESPONSE LENGTH
# =========================================================

def get_max_predict(question: str) -> int:
    q = question.lower()

    if re.search(r"\b\d+\s*words?\b", q):
        return 700

    # Long user questions normally need more room.
    if len(question) > 900:
        return 700

    if len(question) > 500:
        return 500

    return 350


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
async def chat(data: dict):
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"reply": "Please enter a question."}
        )

    # Fast fixed answers.
    direct = direct_answer(user_message)

    if direct:
        return StreamingResponse(
            iter([direct]),
            media_type="text/plain; charset=utf-8"
        )

    # Exact basic arithmetic without depending on the model.
    expression = get_math_expression(user_message)

    if expression:
        result = calculate_math(expression)

        if result is not None:
            return StreamingResponse(
                iter([str(result)]),
                media_type="text/plain; charset=utf-8"
            )

    # Current information only.
    web_information = ""

    if needs_web_search(user_message):
        web_information = web_search(user_message)

    prompt = create_prompt(user_message, web_information)
    max_predict = get_max_predict(user_message)

    def generate():
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.15,
                        "num_ctx": 8192,
                        "num_predict": max_predict,
                        "top_k": 30,
                        "top_p": 0.9,
                        "repeat_penalty": 1.10
                    }
                },
                stream=True,
                timeout=180
            )

            response.raise_for_status()

            answer_parts = []
            in_thinking = False

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Ollama/Qwen can send a separate thinking field.
                if result.get("thinking"):
                    continue

                chunk = result.get("response", "")

                if not chunk:
                    if result.get("done"):
                        break
                    continue

                # Handle think tags even when they arrive across chunks.
                lower_chunk = chunk.lower()

                if in_thinking:
                    end_pos = lower_chunk.find("</think>")

                    if end_pos != -1:
                        in_thinking = False
                        chunk = chunk[end_pos + len("</think>"):]
                    else:
                        continue

                start_pos = lower_chunk.find("<think>")

                if start_pos != -1:
                    before = chunk[:start_pos]
                    after = chunk[start_pos + len("<think>"):]

                    end_pos = after.lower().find("</think>")

                    if end_pos != -1:
                        chunk = before + after[end_pos + len("</think>"):]
                    else:
                        chunk = before
                        in_thinking = True

                if chunk:
                    answer_parts.append(chunk)
                    yield chunk

                if result.get("done"):
                    break

            # Nothing useful came back.
            final_answer = clean_answer("".join(answer_parts))

            if not final_answer:
                yield "Sorry, I could not generate an answer."

        except requests.exceptions.ConnectionError:
            yield "❌ Ollama running nahi hai. Pehle Ollama start karein."

        except requests.exceptions.Timeout:
            yield "❌ Response timeout ho gaya. Thoda chhota question try karein."

        except requests.exceptions.RequestException as e:
            print("OLLAMA REQUEST ERROR:", repr(e))
            yield "❌ Ollama se response nahi mil raha."

        except Exception as e:
            print("SAKAI ERROR:", repr(e))
            yield "❌ SakAI se response nahi mil raha."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
