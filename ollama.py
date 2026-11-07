# ============================================================
# AULA-PRATICA- ROBO PROFESSOR INTELIGENTE DE IA
# Machine Learning + Rede Neural MLP + IA Generativa Ollama
# Interface grafica Tkinter + Voz pyttsx3
# ============================================================

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import math
import json
import urllib.request
import urllib.error
import time
import warnings

import pyttsx3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1-PASSO: CONFIGURACOES
# ============================================================

NOME_ROBO = "TITAN" # (Modificação 1: Nome alterado para TITAN)

MODELO_OLLAMA = "llama3.2"

URL_OLLAMA = "http://localhost:11434/api/chat"

LIMITE_HISTORICO = 12

CONFIANCA_MINIMA = 0.70
MARGEM_MINIMA = 0.15

# Filas para comunicacao entre as threads
fila_comandos = queue.Queue()
fila_interface = queue.Queue()
fila_voz = queue.Queue()

encerrando = threading.Event()

historico_conversa = []

# Estado da animacao
acao_atual = "parado"
inicio_acao = time.time()


# ============================================================
# 2 PASSO: BASE DE TREINAMENTO DO MACHINE LEARNING
# ============================================================

frases_treinamento = [

    # SAUDACAO
    "oi",
    "ola",
    "bom dia",
    "boa tarde",
    "boa noite",
    "e ai titan",
    "ola robo",
    "tudo bem",
    "como vai",
    "saudacoes",

    # NOME
    "qual e seu nome",
    "como voce se chama",
    "diga seu nome",
    "quem e voce",
    "seu nome",
    "qual o nome do robo",
    "voce se chama titan",
    "apresente seu nome",

    # IDENTIDADE
    "o que voce e",
    "voce e um robo",
    "explique quem voce e",
    "fale sobre voce",
    "qual sua funcao",
    "voce e inteligente",
    "qual seu objetivo",
    "o que voce consegue fazer",

    # ACENAR
    "acene",
    "acene com a mao",
    "levante a mao",
    "mande um oi",
    "cumprimente",
    "faca um cumprimento",
    "diga ola com a mao",
    "movimente seu braco",

    # DANCAR
    "dance",
    "vamos dancar",
    "faca uma danca",
    "dance para mim",
    "movimente seu corpo",
    "mostre seus movimentos",
    "danca",
    "rebola",

    # GIRAR
    "gire",
    "gire seu corpo",
    "faca um giro",
    "de uma volta",
    "gire para mim",
    "rotacione",
    "faca uma rotacao",

    # PULAR
    "pule",
    "salte",
    "faca um pulo",
    "pule para cima",
    "mostre um salto",
    "salte no ar",

    # ANDAR
    "ande",
    "caminhe",
    "faca uma caminhada",
    "movimente suas pernas",
    "ande para frente",
    "caminhe um pouco",

    # AJUDA (Modificação 4)
    "ajuda",
    "preciso de ajuda",
    "o que voce faz",
    "comandos",
    "quais os comandos",

    # PYTHON (Modificação 5)
    "o que e python",
    "explique python",
    "o que e a linguagem python",

    # FELIZ (Modificação 7)
    "fique feliz",
    "mostre que esta feliz",
    "sorria",
    "expresse felicidade",

    # DESPEDIDA
    "tchau",
    "ate logo",
    "ate mais",
    "adeus",
    "vou embora",
    "encerrar conversa",
    "finalizar",
    "ate a proxima",

    # CONVERSA GERAL
    "explique inteligencia artificial",
    "o que e inteligencia artificial",
    "o que e uma rede neural",
    "explique machine learning",
    "o que e um neuronio",
    "como funciona um algoritmo",
    "fale sobre tecnologia",
    "explique aprendizado de maquina",
    "como funciona uma inteligencia artificial",
    "o que sao dados",
    "explique deep learning",
    "o que e programacao",
    "como funciona um computador",
    "o que e um robo inteligente",
    "como uma rede neural aprende",
    "qual a diferenca entre ia e machine learning",
    "o que e um modelo computacional"
]

intencoes_treinamento = [

    # SAUDACAO - 10
    "saudacao", "saudacao", "saudacao", "saudacao",
    "saudacao", "saudacao", "saudacao", "saudacao",
    "saudacao", "saudacao",

    # NOME - 8
    "nome", "nome", "nome", "nome",
    "nome", "nome", "nome", "nome",

    # IDENTIDADE - 8
    "identidade", "identidade", "identidade", "identidade",
    "identidade", "identidade", "identidade", "identidade",

    # ACENAR - 8
    "acenar", "acenar", "acenar", "acenar",
    "acenar", "acenar", "acenar", "acenar",

    # DANCAR - 8
    "dancar", "dancar", "dancar", "dancar",
    "dancar", "dancar", "dancar", "dancar",

    # GIRAR - 7
    "girar", "girar", "girar", "girar",
    "girar", "girar", "girar",

    # PULAR - 6
    "pular", "pular", "pular",
    "pular", "pular", "pular",

    # ANDAR - 6
    "andar", "andar", "andar",
    "andar", "andar", "andar",

    # AJUDA - 5 (Modificação 4)
    "ajuda", "ajuda", "ajuda", "ajuda", "ajuda",

    # PYTHON - 3 (Modificação 5)
    "python", "python", "python",

    # FELIZ - 4 (Modificação 7)
    "feliz", "feliz", "feliz", "feliz",

    # DESPEDIDA - 8
    "despedida", "despedida", "despedida", "despedida",
    "despedida", "despedida", "despedida", "despedida",

    # CONVERSA - 17
    "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa", "conversa",
    "conversa", "conversa", "conversa", "conversa",
    "conversa"
]


# ============================================================
# 3 PASSO:TREINAMENTO DA REDE NEURAL
# ============================================================

print(f"\nTreinando a inteligencia do {NOME_ROBO}...")

modelo = Pipeline([

    (
        "vetorizador",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),

    (
        "rede_neural",
        MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=42,
            learning_rate_init=0.002,
            early_stopping=False
        )
    )

])

modelo.fit(frases_treinamento, intencoes_treinamento)

print("Treinamento concluido!")
print("Rede neural pronta para classificar intencoes.\n")


# ============================================================
# 4 PASSO CLASSIFICACAO DE INTENCOES
# ============================================================

def identificar_intencao(texto):

    probabilidades = modelo.predict_proba([texto])[0]

    classes = modelo.classes_

    indices = probabilidades.argsort()[::-1]

    indice_principal = indices[0]

    intencao = classes[indice_principal]

    confianca = float(probabilidades[indice_principal])

    segunda_confianca = float(probabilidades[indices[1]])

    margem = confianca - segunda_confianca

    # Verifica se existem palavras reconhecidas pelo modelo
    vetor = modelo.named_steps["vetorizador"].transform([texto])

    palavras_reconhecidas = vetor.nnz

    return (
        intencao,
        confianca,
        margem,
        palavras_reconhecidas
    )


# ============================================================
# 5 PASSO: DETECCAO DE PERGUNTAS
# ============================================================

def eh_pergunta(texto):

    texto = texto.lower().strip()

    palavras_pergunta = [
        "o que",
        "como",
        "qual",
        "quais",
        "quem",
        "quando",
        "onde",
        "porque",
        "por que",
        "explique",
        "defina",
        "fale sobre",
        "me explique",
        "me ensine",
        "para que"
    ]

    if "?" in texto:
        return True

    for palavra in palavras_pergunta:

        if texto.startswith(palavra):
            return True

    return False


# ============================================================
# 6 PASSO: IA GENERATIVA - OLLAMA
# ============================================================

def conversar_com_ia(pergunta):

    global historico_conversa

    instrucao = (
        f"Seu nome e {NOME_ROBO}. Voce e um robo educacional "
        "que conversa em portugues brasileiro. "
        "Responda de maneira clara, didatica e objetiva. "
        "Explique conceitos de inteligencia artificial, "
        "programacao, tecnologia, matematica e ciencias. "
        "Quando necessario, apresente exemplos simples. "
        "Nao invente informacoes. "
        "Use respostas adequadas para estudantes iniciantes."
    )

    mensagens = [
        {
            "role": "system",
            "content": instrucao
        }
    ]

    # Recupera o contexto das conversas anteriores
    mensagens.extend(historico_conversa)

    mensagens.append({
        "role": "user",
        "content": pergunta
    })

    dados = {
        "model": MODELO_OLLAMA,
        "messages": mensagens,
        "stream": False
    }

    dados_json = json.dumps(dados).encode("utf-8")

    requisicao = urllib.request.Request(
        URL_OLLAMA,
        data=dados_json,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            requisicao,
            timeout=120
        ) as resposta:

            resultado = json.loads(
                resposta.read().decode("utf-8")
            )

        texto_resposta = resultado["message"]["content"].strip()

        # Atualiza o historico somente depois de receber a resposta
        historico_conversa.append({
            "role": "user",
            "content": pergunta
        })

        historico_conversa.append({
            "role": "assistant",
            "content": texto_resposta
        })

        # Mantem somente as ultimas mensagens
        historico_conversa = historico_conversa[-LIMITE_HISTORICO:]

        return texto_resposta

    except urllib.error.URLError:

        return (
            "Nao consegui acessar o Ollama. "
            "Verifique se o programa esta instalado, "
            "se o servico esta ativo e se o modelo llama3.2 "
            "foi baixado."
        )

    except TimeoutError:

        return (
            "A resposta demorou muito. "
            "Tente fazer uma pergunta mais curta."
        )

    except Exception as erro:

        return f"Ocorreu um erro na IA generativa: {erro}"


# ============================================================
# 7 PASSO: RESPOSTAS E ACOES DO ROBO
# ============================================================

def executar_intencao(intencao):

    respostas = {

        "saudacao": (
            f"Ola! Seja bem-vindo. "
            f"Eu sou o {NOME_ROBO}, seu robo educacional."
        ),

        "nome": (
            f"Meu nome e {NOME_ROBO}. "
            "Sou um robo educacional com inteligencia artificial."
        ),

        "identidade": (
            f"Sou o {NOME_ROBO}. "
            "Utilizo aprendizado de maquina, uma rede neural "
            "e inteligencia artificial generativa."
        ),

        "acenar": (
            "Ola! Estou acenando para voce!"
        ),

        "dancar": (
            "Vamos dancar! Movimentando os dois bracos!"
        ),

        "girar": (
            "Preparado? Vou fazer um giro!"
        ),

        "pular": (
            "Olha o meu salto!"
        ),

        "andar": (
            "Vou caminhar um pouco!"
        ),

        # Modificação 4: Resposta para Ajuda
        "ajuda": (
            "Posso realizar acoes como acenar, dancar, girar, pular e andar. "
            "Também posso responder perguntas sobre Inteligência Artificial, Python e tecnologia!"
        ),

        # Modificação 5: Resposta para Python
        "python": (
            "Python é uma linguagem de programação de alto nível, muito "
            "utilizada em Inteligência Artificial, ciência de dados e "
            "desenvolvimento web devido à sua sintaxe simples e clara."
        ),

        # Modificação 7: Resposta para Feliz
        "feliz": (
            "Estou muito feliz em ajudar você a aprender programação e inteligência artificial!"
        ),

        "despedida": (
            "Ate logo! Foi um prazer conversar com voce."
        )
    }

    acoes = {

        "saudacao": "acenar",
        "nome": "parado",
        "identidade": "pensar",
        "acenar": "acenar",
        "dancar": "dancar",
        "girar": "girar",
        "pular": "pular",
        "andar": "andar",
        "ajuda": "parado",
        "python": "pensar",
        "feliz": "feliz", # Modificação 7
        "despedida": "parado"
    }

    return (
        respostas.get(intencao, ""),
        acoes.get(intencao, "parado")
    )


# ============================================================
# 8 PASSO: THREAD DE PROCESSAMENTO
# ============================================================

def processar_comandos():

    while not encerrando.is_set():

        try:

            pergunta = fila_comandos.get(timeout=0.2)

        except queue.Empty:

            continue

        if pergunta is None:
            break

        try:

            intencao, confianca, margem, palavras = (
                identificar_intencao(pergunta)
            )

            # Perguntas e frases desconhecidas vao para a IA generativa
            usar_ia_generativa = (

                intencao == "conversa"

                or eh_pergunta(pergunta)

                or palavras == 0

                or confianca < CONFIANCA_MINIMA

                or margem < MARGEM_MINIMA
            )

            if usar_ia_generativa:

                fila_interface.put((
                    "status",
                    f"{NOME_ROBO} esta consultando a IA generativa..."
                ))

                resposta = conversar_com_ia(pergunta)

                acao = "pensar"

                origem = "IA GENERATIVA"

            else:

                resposta, acao = executar_intencao(intencao)

                origem = "MACHINE LEARNING"

            fila_interface.put((
                "diagnostico",
                intencao,
                confianca,
                origem
            ))

            fila_interface.put((
                "resposta",
                resposta,
                acao
            ))

        except Exception as erro:

            fila_interface.put((
                "resposta",
                f"Erro no processamento: {erro}",
                "parado"
            ))

        finally:

            fila_comandos.task_done()


# ============================================================
# 9 PASSO: THREAD DE VOZ
# ============================================================

def executar_voz():

    motor = None

    try:

        motor = pyttsx3.init()

        motor.setProperty("rate", 190) # (Modificação 2: Velocidade da fala alterada de 160 para 190)
        motor.setProperty("volume", 1.0)

        # Procura uma voz em portugues, se estiver instalada
        vozes = motor.getProperty("voices")

        for voz in vozes:

            dados_voz = (
                voz.id + " " + voz.name
            ).lower()

            if (
                "brazil" in dados_voz
                or "portuguese" in dados_voz
                or "brasil" in dados_voz
                or "portugues" in dados_voz
            ):

                motor.setProperty("voice", voz.id)
                break

    except Exception as erro:

        print("Erro ao inicializar a voz:", erro)

    while not encerrando.is_set():

        try:

            texto = fila_voz.get(timeout=0.2)

        except queue.Empty:

            continue

        if texto is None:
            break

        if motor is not None:

            try:

                motor.say(texto)
                motor.runAndWait()

            except Exception as erro:

                print("Erro na fala:", erro)

        fila_voz.task_done()

    if motor is not None:

        try:
            motor.stop()
        except Exception:
            pass


# ============================================================
# 10 PASSO: INTERFACE GRAFICA
# ============================================================

janela = tk.Tk()

janela.title(f"{NOME_ROBO} - Robo Educacional Inteligente")

janela.geometry("900x720")

janela.configure(bg="#06111D")

janela.resizable(False, False)

# Cabecalho
titulo = tk.Label(
    janela,
    text=NOME_ROBO,
    font=("Arial", 25, "bold"),
    fg="#00C8FF",
    bg="#101820"
)

titulo.pack(pady=10)

subtitulo = tk.Label(
    janela,
    text="Machine Learning | Rede Neural | IA Generativa",
    font=("Arial", 12),
    fg="white",
    bg="#101820"
)

subtitulo.pack()


# ============================================================
# 11 PASSO: AREA DE DESENHO DO ROBO
# ============================================================

canvas = tk.Canvas(
    janela,
    width=640,
    height=380,
    bg="#182632",
    highlightthickness=2,
    highlightbackground="#00AACC"
)

canvas.pack(pady=10)

# Chao
canvas.create_line(
    80, 345, 560, 345,
    fill="#536878",
    width=3
)

canvas.create_oval(
    190, 325, 450, 365,
    fill="#263D4A",
    outline=""
)

# ============================================================
# 12 PASSO: DESENHO DO ROBO METALICO
# ============================================================

# Antena
antena = canvas.create_line(
    320, 40, 320, 70,
    fill="#E2DDEA",
    width=5
)

antena_luz = canvas.create_oval(
    311, 27, 329, 45,
    fill="#FF0000",
    outline="#BDEEFF",
    width=2
)

# Cabeca
cabeca = canvas.create_oval(
    255, 65, 385, 175,
    fill="#2314C4",
    outline="#9F9FA1",
    width=4
)

# Visor
visor = canvas.create_oval(
    267, 90, 373, 151,
    fill="#172B3A",
    outline="#DCE5EA",
    width=3
)

# Olhos (Modificação 3: Cor alterada de azul "#00C8FF" para verde "#00FF66")
olho_esquerdo = canvas.create_oval(
    286, 108, 307, 132,
    fill="#00FF66",
    outline="#A9FFCC",
    width=2
)

olho_direito = canvas.create_oval(
    333, 108, 354, 132,
    fill="#00FF66",
    outline="#A9FFCC",
    width=2
)

# Boca
boca = canvas.create_line(
    303, 145, 337, 145,
    fill="#FFFFFF",
    width=3
)

# Pescoco
pescoco = canvas.create_rectangle(
    303, 174, 337, 194,
    fill="#0D1094",
    outline="#DDE5EA",
    width=2
)

# Tronco
tronco = canvas.create_polygon(
    275, 190,
    365, 190,
    385, 285,
    255, 285,
    fill="#2305C9",
    outline="#F5F8FA",
    width=4
)

# Painel no peito
painel = canvas.create_rectangle(
    293, 215, 347, 263,
    fill="#263D4A",
    outline="#DCE5EA",
    width=3
)

# Luz do peito
luz_peito = canvas.create_oval(
    310, 225, 330, 245,
    fill="#00BFFF",
    outline="#BDEEFF",
    width=2
)

# Bracos
braco_esquerdo = canvas.create_line(
    275, 205, 235, 270,
    fill="#C8D1D8",
    width=17
)

braco_direito = canvas.create_line(
    365, 205, 405, 270,
    fill="#C8D1D8",
    width=17
)

# Juntas
junta_esquerda = canvas.create_oval(
    225, 258, 245, 278,
    fill="#0D0D0E",
    outline="#E5EDF1",
    width=2
)

junta_direita = canvas.create_oval(
    395, 258, 415, 278,
    fill="#0F1011",
    outline="#E5EDF1",
    width=2
)

# Pernas
perna_esquerda = canvas.create_line(
    290, 282, 280, 325,
    fill="#F1F7F7",
    width=18
)

perna_direita = canvas.create_line(
    350, 282, 360, 325,
    fill="#FCF8F8",
    width=18
)

# Pes
pe_esquerdo = canvas.create_oval(
    255, 316, 295, 338,
    fill="#030303",
    outline="#E5EDF1",
    width=3
)

pe_direito = canvas.create_oval(
    345, 316, 385, 338,
    fill="#080808",
    outline="#E5EDF1",
    width=3
)

# ============================================================
# 13 PASSO: CONTROLE DA ANIMACAO
# ============================================================

def definir_acao(nova_acao):

    global acao_atual, inicio_acao

    acao_atual = nova_acao

    inicio_acao = time.time()


def atualizar_robo():

    global acao_atual

    tempo = time.time() - inicio_acao

    # Posicao padrao
    centro_x = 320
    deslocamento_y = 0

    braco_esq_x = 235
    braco_esq_y = 270

    braco_dir_x = 405
    braco_dir_y = 270

    perna_esq_x = 280
    perna_dir_x = 360

    # Modificação 8: Amplitude modificada de 15 para 25 pixels nas variáveis de movimento abaixo onde aplicável
    # MOVIMENTO DE PULO
    if acao_atual == "pular":

        deslocamento_y = -abs(math.sin(tempo * 5)) * 40

        if tempo > 1.5:
            definir_acao("parado")

    # MOVIMENTO DE DANCAR (Modificação 6: Movimenta os dois braços simultaneamente)
    elif acao_atual == "dancar":

        deslocamento_y = math.sin(tempo * 8) * 8

        braco_esq_x = 235 - math.sin(tempo * 10) * 25
        braco_esq_y = 270 - abs(math.sin(tempo * 10)) * 35

        braco_dir_x = 405 + math.sin(tempo * 10) * 25
        braco_dir_y = 270 - abs(math.sin(tempo * 10)) * 35

        perna_esq_x = 280 + math.sin(tempo * 8) * 25
        perna_dir_x = 360 - math.sin(tempo * 8) * 25

        if tempo > 5:
            definir_acao("parado")

    # MOVIMENTO DE ACENAR
    elif acao_atual == "acenar":

        braco_dir_x = 405 + math.sin(tempo * 12) * 25
        braco_dir_y = 200 + math.sin(tempo * 12) * 25

        if tempo > 2.5:
            definir_acao("parado")

    # MOVIMENTO DE ANDAR
    elif acao_atual == "andar":

        braco_esq_x = 235 + math.sin(tempo * 8) * 25
        braco_dir_x = 405 - math.sin(tempo * 8) * 25

        perna_esq_x = 280 + math.sin(tempo * 8) * 25
        perna_dir_x = 360 - math.sin(tempo * 8) * 25

        if tempo > 4:
            definir_acao("parado")

    # MOVIMENTO DE GIRAR
    elif acao_atual == "girar":

        escala = abs(math.cos(tempo * 4))

        largura = 130 * escala

        canvas.coords(
            cabeca,
            centro_x - largura / 2,
            65,
            centro_x + largura / 2,
            175
        )

        if tempo > 2:
            definir_acao("parado")

    # MOVIMENTO DE FELIZ (Modificação 7: Estado novo com leve pulo/balanço alegre)
    elif acao_atual == "feliz":

        deslocamento_y = -abs(math.sin(tempo * 10)) * 25

        braco_esq_x = 235 - 25
        braco_esq_y = 270 - 45

        braco_dir_x = 405 + 25
        braco_dir_y = 270 - 45

        if tempo > 3:
            definir_acao("parado")

    else:

        # Movimento natural de repouso
        deslocamento_y = math.sin(tempo * 2) * 3

        # Restaura largura normal da cabeca
        canvas.coords(
            cabeca,
            255, 65, 385, 175
        )

    # ========================================================
    # ATUALIZACAO DAS COORDENADAS
    # ========================================================

    y = deslocamento_y

    canvas.coords(antena, 320, 40+y, 320, 70+y)
    canvas.coords(antena_luz, 311, 27+y, 329, 45+y)

    if acao_atual != "girar":

        canvas.coords(
            cabeca,
            255, 65+y, 385, 175+y
        )

    canvas.coords(visor, 267, 90+y, 373, 151+y)

    canvas.coords(olho_esquerdo, 286, 108+y, 307, 132+y)
    canvas.coords(olho_direito, 333, 108+y, 354, 132+y)

    canvas.coords(boca, 303, 145+y, 337, 145+y)

    canvas.coords(pescoco, 303, 174+y, 337, 194+y)

    canvas.coords(
        tronco,
        275, 190+y,
        365, 190+y,
        385, 285+y,
        255, 285+y
    )

    canvas.coords(painel, 293, 215+y, 347, 263+y)
    canvas.coords(luz_peito, 310, 225+y, 330, 245+y)

    canvas.coords(
        braco_esquerdo,
        275, 205+y,
        braco_esq_x, braco_esq_y+y
    )

    canvas.coords(
        braco_direito,
        365, 205+y,
        braco_dir_x, braco_dir_y+y
    )

    canvas.coords(
        junta_esquerda,
        braco_esq_x-10, braco_esq_y-10+y,
        braco_esq_x+10, braco_esq_y+10+y
    )

    canvas.coords(
        junta_direita,
        braco_dir_x-10, braco_dir_y-10+y,
        braco_dir_x+10, braco_dir_y+10+y
    )

    canvas.coords(
        perna_esquerda,
        290, 282+y,
        perna_esq_x, 325+y
    )

    canvas.coords(
        perna_direita,
        350, 282+y,
        perna_dir_x, 325+y
    )

    canvas.coords(
        pe_esquerdo,
        perna_esq_x-25, 316+y,
        perna_esq_x+15, 338+y
    )

    canvas.coords(
        pe_direito,
        perna_dir_x-15, 316+y,
        perna_dir_x+25, 338+y
    )

    janela.after(50, atualizar_robo)


# ============================================================
# 14 PASSO: AREA DE CONVERSA
# ============================================================

painel_conversa = scrolledtext.ScrolledText(
    janela,
    width=95,
    height=9,
    font=("Consolas", 10),
    bg="#0C141B",
    fg="#E6F4F8",
    insertbackground="white",
    wrap=tk.WORD
)

painel_conversa.pack(padx=15, pady=8)

painel_conversa.config(state=tk.DISABLED)


def escrever_mensagem(remetente, mensagem):

    painel_conversa.config(state=tk.NORMAL)

    painel_conversa.insert(
        tk.END,
        f"{remetente}: {mensagem}\n\n"
    )

    painel_conversa.see(tk.END)

    painel_conversa.config(state=tk.DISABLED)


# ============================================================
# 15 PASSO: ENTRADA DE COMANDOS
# ============================================================

entrada = tk.Entry(
    janela,
    font=("Arial", 13),
    width=65,
    bg="#E8F1F5",
    fg="#101820"
)

entrada.pack(pady=8)


def enviar_comando():

    pergunta = entrada.get().strip()

    if not pergunta:
        return

    if pergunta.lower() in ["sair", "fechar", "encerrar programa"]:

        escrever_mensagem(
            NOME_ROBO,
            "Ate logo! Encerrando o sistema."
        )

        fechar_programa()
        return

    escrever_mensagem("VOCE", pergunta)

    fila_interface.put((
        "status",
        f"{NOME_ROBO} esta processando sua mensagem..."
    ))

    fila_comandos.put(pergunta)

    entrada.delete(0, tk.END)


botao_enviar = tk.Button(
    janela,
    text="ENVIAR COMANDO",
    command=enviar_comando,
    font=("Arial", 11, "bold"),
    bg="#007FA3",
    fg="white",
    activebackground="#00AACC",
    width=22
)

botao_enviar.pack(pady=5)

entrada.bind("<Return>", lambda evento: enviar_comando())


# ============================================================
# 16 PASSO: STATUS DO ROBO
# ============================================================

status = tk.Label(
    janela,
    text=f"{NOME_ROBO} pronto para conversar.",
    font=("Arial", 10, "bold"),
    fg="#00D5FF",
    bg="#101820"
)

status.pack(pady=5)


# ============================================================
# 17 PASSO: RECEBIMENTO DAS RESPOSTAS NA INTERFACE
# ============================================================

def verificar_fila_interface():

    while True:

        try:

            evento = fila_interface.get_nowait()

        except queue.Empty:

            break

        tipo = evento[0]

        if tipo == "status":

            status.config(text=evento[1])

        elif tipo == "diagnostico":

            intencao = evento[1]
            confianca = evento[2]
            origem = evento[3]

            status.config(
                text=(
                    f"Origem: {origem} | "
                    f"Intencao: {intencao} | "
                    f"Probabilidade: {confianca:.2f}"
                )
            )

        elif tipo == "resposta":

            resposta = evento[1]
            acao = evento[2]

            escrever_mensagem(NOME_ROBO, resposta)

            definir_acao(acao)

            # Envia a fala para a thread de voz
            fila_voz.put(resposta)

    if not encerrando.is_set():

        janela.after(100, verificar_fila_interface)


# ============================================================
# 18 PASSO: ENCERRAMENTO
# ============================================================

def fechar_programa():

    if encerrando.is_set():
        return

    encerrando.set()

    fila_comandos.put(None)
    fila_voz.put(None)

    try:
        janela.destroy()
    except tk.TclError:
        pass


janela.protocol(
    "WM_DELETE_WINDOW",
    fechar_programa
)


# ============================================================
# 19 PASSO: INICIALIZACAO DAS THREADS
# ============================================================

thread_processamento = threading.Thread(
    target=processar_comandos,
    daemon=True
)

thread_voz = threading.Thread(
    target=executar_voz,
    daemon=True
)

thread_processamento.start()
thread_voz.start()


# ============================================================
# 20 PASSO: INICIO DO SISTEMA
# ============================================================

escrever_mensagem(
    NOME_ROBO,
    f"Ola! Eu sou o {NOME_ROBO}. "
    "Minha inteligencia artificial esta pronta!"
)

fila_voz.put(
    f"Ola! Eu sou o Titan. "
    "Minha inteligencia artificial esta pronta!"
)

atualizar_robo()

verificar_fila_interface()

janela.mainloop()