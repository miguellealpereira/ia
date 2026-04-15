import tkinter as tk
from tkinter import simpledialog, messagebox
import json
import os
import re
import random
import threading
import traceback
import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================
# CONFIG
# =============================
ARQUIVO_APRENDIZADO = "aprendizado.json"
ARQUIVO_MEMORIA = "memoria.json"
ARQUIVO_HISTORICO = "historico.json"
SEED = 42

# Para testar sem travar por causa da busca web,
# deixe False primeiro. Depois mude para True.
USAR_BUSCA_ONLINE = False


np.random.seed(SEED)

# =============================
# FUNÇÕES JSON
# =============================
def carregar_json_seguro(caminho, padrao):
    if not os.path.exists(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(padrao, f, ensure_ascii=False, indent=4)
        return padrao

    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return padrao


def salvar_json_seguro(caminho, dados):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)


# =============================
# MEMÓRIA PERSISTENTE
# =============================
memoria = carregar_json_seguro(
    ARQUIVO_MEMORIA,
    {
        "nome": None,
        "ultimo_assunto": None,
        "preferencias": {}
    }
)

historico = carregar_json_seguro(ARQUIVO_HISTORICO, [])


# =============================
# CORREÇÕES DE PORTUGUÊS
# =============================
correcoes = {
    "vc": "você",
    "voce": "você",
    "tbm": "também",
    "tambem": "também",
    "nao": "não",
    "pq": "porque",
    "q": "que",
    "to": "estou",
    "tô": "estou",
    "ta": "está",
    "tá": "está",
    "blz": "beleza",
    "oiii": "oi",
    "oii": "oi",
    "oie": "oi",
    "ola": "olá",
    "flw": "falou",
    "ansioza": "ansiosa",
    "tristee": "triste",
    "cansadaa": "cansada",
    "foca": "focar",
    "história": "historia",
    "neoral": "neural",
    "nerual": "neural",
    "carros": "carro",
    "oq": "o que",
    "eh": "é"
}


# =============================
# BASE FIXA
# =============================
base_fixa = [
    {
        "tag": "saudacao",
        "padroes": [
            "oi", "olá", "opa", "bom dia", "boa tarde", "boa noite",
            "ola", "oiii", "oii", "oie"
        ],
        "respostas": [
            "Oi{nome}! Como posso te ajudar?",
            "Olá{nome}! Estou pronta para conversar.",
            "Opa{nome}! Me diz no que você precisa."
        ]
    },
    {
        "tag": "despedida",
        "padroes": ["tchau", "até mais", "falou", "adeus", "até logo", "flw"],
        "respostas": [
            "Tchau{nome}! Até a próxima.",
            "Até mais{nome}!",
            "Foi bom conversar com você{nome}."
        ]
    },
    {
        "tag": "emocional",
        "padroes": [
            "estou triste", "não estou bem", "nao estou bem", "me sinto mal",
            "estou cansada", "tô desanimada", "to desanimada", "estou ansiosa",
            "to triste", "to mal"
        ],
        "respostas": [
            "Quer conversar sobre isso? Estou aqui para te ouvir.",
            "Sinto muito que você esteja assim. Posso ficar com você nessa conversa.",
            "Talvez ajude respirar um pouco e me contar o que aconteceu."
        ]
    },
    {
        "tag": "estudos",
        "padroes": [
            "preciso estudar", "me ajuda a estudar", "como estudar melhor",
            "dicas de estudo", "não consigo focar", "nao consigo focar",
            "como aprender mais rápido", "como aprender mais rapido",
            "quero estudar", "preciso focar"
        ],
        "respostas": [
            "Posso te ajudar a montar um plano de estudos.",
            "Uma boa estratégia é estudar em blocos curtos com pausas.",
            "Posso te passar dicas para foco e revisão."
        ]
    },
    {
        "tag": "historia",
        "padroes": [
            "me ensine historia", "me ajuda com historia", "quero estudar historia",
            "preciso estudar historia", "historia do brasil", "me explique historia",
            "quero aprender historia", "ensina historia", "estudar historia"
        ],
        "respostas": [
            "Posso te ajudar com história. Qual assunto você quer estudar?",
            "Claro! Me diga qual tema de história você quer aprender.",
            "História é um tema amplo. Quer Brasil, Geral ou algum período específico?"
        ]
    }
]


# =============================
# TEXTO
# =============================
def corrigir_texto(texto):
    palavras = texto.lower().split()
    corrigidas = [correcoes.get(p, p) for p in palavras]
    return " ".join(corrigidas)


def preprocessar_texto(texto):
    texto = texto.lower().strip()
    texto = corrigir_texto(texto)
    texto = re.sub(r"[^a-záàâãéèêíïóôõöúçñ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extrair_termo_busca(texto):
    texto = preprocessar_texto(texto)

    remover = [
        "o que é ", "o que e ", "o que um ", "o que uma ", "o que o ", "o que a ",
        "o que faz ", "para que serve ", "pra que serve ", "qual a função de ",
        "qual a funcao de ", "quem é ", "quem e ", "quem foi ", "me fale sobre ",
        "me fala sobre ", "quero saber sobre ", "me explique ", "me explica ",
        "ensine ", "me ensine ", "pode me explicar ", "pode explicar ",
        "quero aprender sobre "
    ]

    for r in remover:
        if texto.startswith(r):
            texto = texto[len(r):].strip()
            break

    texto = re.sub(r"\b(faz|serve|funciona)\b", "", texto).strip()

    stopwords = ["um", "uma", "de", "do", "da", "o", "a", "os", "as", "sobre"]
    palavras = [p for p in texto.split() if p not in stopwords]

    return " ".join(palavras).strip()


# =============================
# BUSCA WEB
# =============================
def buscar_titulo_wikipedia(termo):
    try:
        url = "https://pt.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": termo,
            "format": "json",
            "utf8": 1,
            "srlimit": 1
        }

        resposta = requests.get(
            url,
            params=params,
            timeout=5,
            headers={"User-Agent": "MeuChatbotIA/1.0"}
        )

        if resposta.status_code != 200:
            return None

        dados = resposta.json()
        resultados = dados.get("query", {}).get("search", [])
        return resultados[0]["title"] if resultados else None
    except Exception:
        return None


def buscar_resumo_wikipedia_por_titulo(titulo):
    try:
        if not titulo:
            return None

        titulo = titulo.replace(" ", "_")
        url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{titulo}"

        resposta = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "MeuChatbotIA/1.0"}
        )

        if resposta.status_code != 200:
            return None

        dados = resposta.json()
        resumo = dados.get("extract")
        return resumo.strip() if resumo else None
    except Exception:
        return None


def buscar_duckduckgo(query):
    try:
        url = "https://html.duckduckgo.com/html/"
        resposta = requests.post(
            url,
            data={"q": query},
            timeout=6,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        if resposta.status_code != 200:
            return []

        soup = BeautifulSoup(resposta.text, "html.parser")
        resultados = []

        for a in soup.select("a.result__a")[:3]:
            href = a.get("href", "").strip()
            titulo = a.get_text(" ", strip=True)
            if href and titulo:
                resultados.append({"titulo": titulo, "url": href})

        return resultados
    except Exception:
        return []


def extrair_texto_pagina(url):
    try:
        resposta = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if resposta.status_code != 200:
            return None

        if "text/html" not in resposta.headers.get("Content-Type", ""):
            return None

        soup = BeautifulSoup(resposta.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        texto = soup.get_text(" ", strip=True)
        texto = re.sub(r"\s+", " ", texto).strip()

        if len(texto) < 200:
            return None

        return texto[:10000]
    except Exception:
        return None


def dividir_frases(texto):
    frases = re.split(r"(?<=[.!?])\s+", texto)
    return [f.strip() for f in frases if len(f.strip()) > 30]


def resumir_texto_relevante(texto, pergunta, limite_frases=4):
    frases = dividir_frases(texto)
    if not frases:
        return None

    termos_pergunta = set(preprocessar_texto(pergunta).split())
    pontuadas = []

    for frase in frases:
        frase_limpa = preprocessar_texto(frase)
        termos_frase = set(frase_limpa.split())
        intersecao = len(termos_pergunta & termos_frase)
        bonus_tamanho = min(len(frase) / 200, 1.0)
        score = intersecao + bonus_tamanho

        if score > 0:
            pontuadas.append((score, frase))

    if not pontuadas:
        return " ".join(frases[:2])[:700]

    pontuadas.sort(key=lambda x: x[0], reverse=True)

    melhores = []
    usadas = set()

    for _, frase in pontuadas:
        chave = preprocessar_texto(frase)
        if chave not in usadas:
            melhores.append(frase)
            usadas.add(chave)
        if len(melhores) >= limite_frases:
            break

    resumo = " ".join(melhores).strip()
    if len(resumo) > 700:
        resumo = resumo[:700].rsplit(" ", 1)[0] + "..."
    return resumo


def buscar_online(texto):
    termo = extrair_termo_busca(texto)
    if not termo:
        return None

    try:
        resultados = buscar_duckduckgo(termo)

        for resultado in resultados[:2]:
            conteudo = extrair_texto_pagina(resultado["url"])
            if not conteudo:
                continue

            resumo = resumir_texto_relevante(conteudo, termo)
            if resumo:
                return f"{resumo}\n\nFonte: {resultado['titulo']}"
    except Exception:
        pass

    try:
        titulo = buscar_titulo_wikipedia(termo)
        resumo_wiki = buscar_resumo_wikipedia_por_titulo(titulo)

        if resumo_wiki:
            if len(resumo_wiki) > 700:
                resumo_wiki = resumo_wiki[:700].rsplit(" ", 1)[0] + "..."
            return f"{resumo_wiki}\n\nFonte: Wikipédia"
    except Exception:
        pass

    return None


# =============================
# APRENDIZADO
# =============================
def garantir_arquivo_aprendizado():
    if not os.path.exists(ARQUIVO_APRENDIZADO):
        with open(ARQUIVO_APRENDIZADO, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)


def carregar_aprendizado():
    garantir_arquivo_aprendizado()
    with open(ARQUIVO_APRENDIZADO, "r", encoding="utf-8") as f:
        dados = json.load(f)

    if not isinstance(dados, list):
        return []

    base_valida = []
    for item in dados:
        if (
            isinstance(item, dict)
            and "tag" in item
            and "padroes" in item
            and "respostas" in item
            and isinstance(item["padroes"], list)
            and isinstance(item["respostas"], list)
        ):
            base_valida.append(item)

    return base_valida


def salvar_aprendizado(frase_usuario, resposta_certa, tag="aprendido"):
    frase_usuario = preprocessar_texto(frase_usuario)
    resposta_certa = resposta_certa.strip()

    if not frase_usuario or not resposta_certa:
        return

    dados = carregar_aprendizado()

    for item in dados:
        if item["tag"] == tag:
            if frase_usuario not in item["padroes"]:
                item["padroes"].append(frase_usuario)
            if resposta_certa not in item["respostas"]:
                item["respostas"].append(resposta_certa)
            break
    else:
        dados.append({
            "tag": tag,
            "padroes": [frase_usuario],
            "respostas": [resposta_certa]
        })

    salvar_json_seguro(ARQUIVO_APRENDIZADO, dados)


# =============================
# REDE NEURAL AVANÇADA
# =============================
class RedeNeuralAvancada:
    def __init__(self, entrada_dim, saida_dim, hidden_dims=(32, 16), dropout=0.05, lr=0.03, l2=1e-4, seed=42):
        self.dropout = dropout
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(seed)

        dims = [entrada_dim, *hidden_dims, saida_dim]
        self.pesos = []
        self.vieses = []

        for i in range(len(dims) - 1):
            fan_in = dims[i]
            W = self.rng.normal(0, np.sqrt(2.0 / fan_in), size=(dims[i], dims[i + 1])).astype(np.float32)
            b = np.zeros((1, dims[i + 1]), dtype=np.float32)
            self.pesos.append(W)
            self.vieses.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivada(self, x):
        return (x > 0).astype(np.float32)

    def softmax(self, x):
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, treino=False):
        ativacoes = [X]
        zs = []
        dropout_masks = []

        a = X
        for i in range(len(self.pesos) - 1):
            z = np.dot(a, self.pesos[i]) + self.vieses[i]
            a = self.relu(z)

            if treino and self.dropout > 0:
                mask = (self.rng.random(a.shape) > self.dropout).astype(np.float32)
                a = a * mask / (1.0 - self.dropout)
            else:
                mask = np.ones_like(a, dtype=np.float32)

            zs.append(z)
            ativacoes.append(a)
            dropout_masks.append(mask)

        z_out = np.dot(a, self.pesos[-1]) + self.vieses[-1]
        y_pred = self.softmax(z_out)
        zs.append(z_out)
        ativacoes.append(y_pred)

        return ativacoes, zs, dropout_masks

    def backward(self, y_true, ativacoes, zs, dropout_masks):
        m = y_true.shape[0]
        grad_w = [None] * len(self.pesos)
        grad_b = [None] * len(self.vieses)

        delta = (ativacoes[-1] - y_true) / m
        grad_w[-1] = np.dot(ativacoes[-2].T, delta) + self.l2 * self.pesos[-1]
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.pesos) - 2, -1, -1):
            delta = np.dot(delta, self.pesos[i + 1].T)
            delta = delta * self.relu_derivada(zs[i])
            delta = delta * dropout_masks[i]
            grad_w[i] = np.dot(ativacoes[i].T, delta) + self.l2 * self.pesos[i]
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def step(self, grad_w, grad_b):
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.lr * grad_w[i]
            self.vieses[i] -= self.lr * grad_b[i]

    def treinar(self, X, y, epocas=250, batch_size=8):
        n = X.shape[0]
        indices = np.arange(n)

        for _ in range(epocas):
            self.rng.shuffle(indices)
            X_shuf = X[indices]
            y_shuf = y[indices]

            for inicio in range(0, n, batch_size):
                fim = inicio + batch_size
                xb = X_shuf[inicio:fim]
                yb = y_shuf[inicio:fim]

                ativacoes, zs, dropout_masks = self.forward(xb, treino=True)
                grad_w, grad_b = self.backward(yb, ativacoes, zs, dropout_masks)
                self.step(grad_w, grad_b)

    def prever_probabilidades(self, X):
        ativacoes, _, _ = self.forward(X, treino=False)
        return ativacoes[-1]


# =============================
# TREINO
# =============================
def montar_base_completa():
    base = list(base_fixa)
    base.extend(carregar_aprendizado())
    return base


def construir_dados(base_treino):
    frases = []
    y_treino = []
    classes = sorted(list(set(item["tag"] for item in base_treino)))

    for item in base_treino:
        for padrao in item["padroes"]:
            frase = preprocessar_texto(padrao)
            frases.append(frase)

            saida = [0.0] * len(classes)
            saida[classes.index(item["tag"])] = 1.0
            y_treino.append(saida)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_treino = vectorizer.fit_transform(frases).toarray().astype(np.float32)
    y_treino = np.array(y_treino, dtype=np.float32)

    return vectorizer, classes, X_treino, y_treino


def recriar_modelo():
    base_treino = montar_base_completa()
    vectorizer, classes, X_treino, y_treino = construir_dados(base_treino)

    rede = RedeNeuralAvancada(
        entrada_dim=X_treino.shape[1],
        saida_dim=len(classes),
        hidden_dims=(32, 16),
        dropout=0.05,
        lr=0.03,
        l2=1e-4,
        seed=SEED
    )
    rede.treinar(X_treino, y_treino, epocas=250, batch_size=8)

    return rede, base_treino, vectorizer, classes


# =============================
# RESPOSTA
# =============================
def detectar_nome(texto):
    texto = preprocessar_texto(texto)

    gatilhos = [
        "meu nome é ",
        "meu nome e ",
        "eu me chamo ",
        "me chamo ",
        "pode me chamar de "
    ]

    for gatilho in gatilhos:
        if texto.startswith(gatilho):
            nome = texto[len(gatilho):].strip().title()
            if nome:
                return nome

    return None


def escolher_resposta(item):
    return random.choice(item["respostas"])


def atualizar_historico(msg_usuario, resposta_bot):
    historico.append({"usuario": msg_usuario, "bot": resposta_bot})
    while len(historico) > 20:
        historico.pop(0)
    salvar_json_seguro(ARQUIVO_HISTORICO, historico)
    salvar_json_seguro(ARQUIVO_MEMORIA, memoria)


def responder(texto, rede, base_treino, vectorizer, classes):
    texto_limpo = preprocessar_texto(texto)

    nome = detectar_nome(texto_limpo)
    if nome:
        memoria["nome"] = nome
        salvar_json_seguro(ARQUIVO_MEMORIA, memoria)
        resposta_nome = f"Prazer, {nome}! Vou lembrar de você."
        atualizar_historico(texto, resposta_nome)
        return resposta_nome

    entrada = vectorizer.transform([texto_limpo]).toarray().astype(np.float32)
    previsao = rede.prever_probabilidades(entrada)[0]

    limite = 0.30
    idx = int(np.argmax(previsao))
    confianca = float(previsao[idx])

    if confianca < limite:
        if USAR_BUSCA_ONLINE:
            resposta_online = buscar_online(texto_limpo)

            if resposta_online:
                memoria["ultimo_assunto"] = "internet"
                salvar_json_seguro(ARQUIVO_MEMORIA, memoria)
                resposta = f"Encontrei isto na internet:\n\n{resposta_online}"
                atualizar_historico(texto, resposta)
                return resposta

        return None

    tag = classes[idx]
    memoria["ultimo_assunto"] = tag
    salvar_json_seguro(ARQUIVO_MEMORIA, memoria)

    item_encontrado = None
    for item in base_treino:
        if item["tag"] == tag:
            item_encontrado = item
            break

    if item_encontrado is None:
        return None

    resposta_final = escolher_resposta(item_encontrado)

    nome_usuario = memoria.get("nome", "")
    if nome_usuario:
        nome_usuario = ", " + nome_usuario

    resposta_final = resposta_final.replace("{nome}", nome_usuario)
    resposta_final = f"{resposta_final}\n\n(confiança: {confianca:.2f} | assunto: {tag})"

    atualizar_historico(texto, resposta_final)
    return resposta_final


# =============================
# INTERFACE ESCURA
# =============================
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meu Chatbot IA")
        self.root.geometry("860x620")
        self.root.minsize(700, 520)
        self.root.configure(bg="#0f172a")

        self.rede, self.base_treino, self.vectorizer, self.classes = recriar_modelo()

        self.criar_interface()
        self.carregar_historico_na_tela()

        if not historico:
            self.adicionar_bolha("Bot", "Oi! Estou pronta para conversar com você.")

    def criar_interface(self):
        topo = tk.Frame(self.root, bg="#111827", padx=12, pady=12)
        topo.pack(fill="x")

        titulo = tk.Label(
            topo,
            text="Chatbot com aprendizado",
            font=("Arial", 16, "bold"),
            bg="#111827",
            fg="#f9fafb"
        )
        titulo.pack(side="left")

        btn_aprender = tk.Button(
            topo,
            text="Ensinar resposta",
            command=self.abrir_janela_aprendizado,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=6
        )
        btn_aprender.pack(side="right")

        self.canvas = tk.Canvas(self.root, bg="#0f172a", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.container = tk.Frame(self.canvas, bg="#0f172a")

        self.container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=12)
        self.scrollbar.pack(side="right", fill="y", pady=12)

        rodape = tk.Frame(self.root, bg="#111827", padx=12, pady=12)
        rodape.pack(fill="x")

        self.entrada = tk.Entry(
            rodape,
            font=("Arial", 12),
            bg="#1f2937",
            fg="white",
            insertbackground="white",
            relief="flat"
        )
        self.entrada.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=10)
        self.entrada.bind("<Return>", self.enviar_mensagem)

        btn_enviar = tk.Button(
            rodape,
            text="Enviar",
            width=12,
            command=self.enviar_mensagem,
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=8
        )
        btn_enviar.pack(side="right")

    def carregar_historico_na_tela(self):
        for item in historico:
            self.adicionar_bolha("Você", item["usuario"])
            self.adicionar_bolha("Bot", item["bot"])

    def adicionar_bolha(self, autor, mensagem):
        linha = tk.Frame(self.container, bg="#0f172a")
        linha.pack(fill="x", padx=10, pady=6)

        if autor == "Você":
            anchor = "e"
            cor = "#2563eb"
            fg = "white"
        else:
            anchor = "w"
            cor = "#1f2937"
            fg = "#f9fafb"

        bolha = tk.Label(
            linha,
            text=f"{autor}: {mensagem}",
            justify="left",
            wraplength=560,
            font=("Arial", 11),
            bg=cor,
            fg=fg,
            padx=12,
            pady=10
        )
        bolha.pack(anchor=anchor)

        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def remover_ultima_bolha(self):
        filhos = self.container.winfo_children()
        if filhos:
            filhos[-1].destroy()
            self.root.update_idletasks()
            self.canvas.yview_moveto(1.0)

    def enviar_mensagem(self, event=None):
        msg = self.entrada.get().strip()
        if not msg:
            return

        self.entrada.delete(0, tk.END)
        self.adicionar_bolha("Você", msg)
        self.adicionar_bolha("Bot", "Pensando...")
        self.entrada.config(state="disabled")

        def tarefa():
            try:
                if preprocessar_texto(msg) == "sair":
                    def fechar():
                        self.remover_ultima_bolha()
                        resposta = "Até mais!"
                        self.adicionar_bolha("Bot", resposta)
                        atualizar_historico(msg, resposta)
                        self.root.after(600, self.root.destroy)
                    self.root.after(0, fechar)
                    return

                resposta = responder(msg, self.rede, self.base_treino, self.vectorizer, self.classes)

                if resposta is None:
                    def fluxo_sem_resposta():
                        self.remover_ultima_bolha()
                        self.adicionar_bolha("Bot", "Não encontrei uma resposta local nem online.")
                        ensinar = messagebox.askyesno("Aprender", "Quer me ensinar essa resposta agora?")

                        if ensinar:
                            resposta_certa = simpledialog.askstring("Ensinar", "Como eu deveria responder?")
                            if not resposta_certa:
                                self.adicionar_bolha("Bot", "Tudo bem. Você pode me ensinar depois.")
                                self.entrada.config(state="normal")
                                return

                            tag = simpledialog.askstring(
                                "Categoria",
                                "Qual categoria/tag devo usar?\n(Deixe vazio para 'aprendido')"
                            )
                            if not tag or not tag.strip():
                                tag = "aprendido"

                            def retreinar_em_fundo():
                                try:
                                    salvar_aprendizado(msg, resposta_certa, tag.strip().lower())
                                    nova_rede, nova_base, novo_vectorizer, novas_classes = recriar_modelo()

                                    def concluir():
                                        self.rede = nova_rede
                                        self.base_treino = nova_base
                                        self.vectorizer = novo_vectorizer
                                        self.classes = novas_classes
                                        resposta_aprendeu = "Aprendi isso e já atualizei meu treino."
                                        self.adicionar_bolha("Bot", resposta_aprendeu)
                                        atualizar_historico(msg, resposta_aprendeu)
                                        self.entrada.config(state="normal")

                                    self.root.after(0, concluir)

                                except Exception as e:
                                    erro = traceback.format_exc()

                                    def mostrar_erro():
                                        self.adicionar_bolha("Bot", f"Erro ao aprender:\n{e}")
                                        print(erro)
                                        self.entrada.config(state="normal")

                                    self.root.after(0, mostrar_erro)

                            self.adicionar_bolha("Bot", "Atualizando meu treino...")
                            threading.Thread(target=retreinar_em_fundo, daemon=True).start()

                        else:
                            resposta_negada = "Tudo bem. Tenta falar de outro jeito."
                            self.adicionar_bolha("Bot", resposta_negada)
                            atualizar_historico(msg, resposta_negada)
                            self.entrada.config(state="normal")

                    self.root.after(0, fluxo_sem_resposta)

                else:
                    def fluxo_com_resposta():
                        self.remover_ultima_bolha()
                        self.adicionar_bolha("Bot", resposta)
                        self.entrada.config(state="normal")

                    self.root.after(0, fluxo_com_resposta)

            except Exception as e:
                erro = traceback.format_exc()

                def mostrar_erro():
                    self.remover_ultima_bolha()
                    self.adicionar_bolha("Bot", f"Deu erro:\n{e}")
                    print(erro)
                    self.entrada.config(state="normal")

                self.root.after(0, mostrar_erro)

        threading.Thread(target=tarefa, daemon=True).start()

    def abrir_janela_aprendizado(self):
        frase = simpledialog.askstring("Nova frase", "Frase que devo aprender:")
        if not frase:
            return

        tag = simpledialog.askstring("Categoria", "Nome da categoria/tag:")
        if not tag or not tag.strip():
            tag = "aprendido"

        resposta = simpledialog.askstring("Resposta", "Resposta correta:")
        if not resposta:
            return

        self.adicionar_bolha("Bot", "Atualizando meu treino...")

        def tarefa_aprender():
            try:
                salvar_aprendizado(frase, resposta, tag.strip().lower())
                nova_rede, nova_base, novo_vectorizer, novas_classes = recriar_modelo()

                def concluir():
                    self.rede = nova_rede
                    self.base_treino = nova_base
                    self.vectorizer = novo_vectorizer
                    self.classes = novas_classes
                    self.adicionar_bolha("Bot", f"Aprendi a categoria '{tag.strip().lower()}'.")

                self.root.after(0, concluir)

            except Exception as e:
                erro = traceback.format_exc()

                def mostrar_erro():
                    self.adicionar_bolha("Bot", f"Erro ao aprender:\n{e}")
                    print(erro)

                self.root.after(0, mostrar_erro)

        threading.Thread(target=tarefa_aprender, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()