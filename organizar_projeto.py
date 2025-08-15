import os
import shutil
from pathlib import Path

# 🎯 Diretórios principais
estrutura = [
    "imagens/carro",
    "imagens/moto",
    "dataset/train/carro",
    "dataset/train/moto",
    "dataset/validation/carro",
    "dataset/validation/moto",
    "modelo",
    "resultados"
]

# 📁 Função para criar pastas
def criar_pastas(lista):
    for pasta in lista:
        Path(pasta).mkdir(parents=True, exist_ok=True)
    print("✅ Estrutura de pastas criada.")

# 📦 Função para mover imagens por padrão
def mover_imagens(origem, destino, padrao):
    arquivos = os.listdir(origem)
    if not arquivos:
        print(f"🚫 Nenhum arquivo encontrado em {origem}")
        return

    for arquivo in arquivos:
        if padrao in arquivo.lower() and arquivo.lower().endswith(('.jpg', '.png')):
            origem_path = os.path.join(origem, arquivo)
            destino_path = os.path.join(destino, arquivo)
            if not os.path.exists(destino_path):
                shutil.move(origem_path, destino_path)
                print(f"📦 Movido: {arquivo} → {destino}")
            else:
                print(f"⚠️ Já existe: {arquivo} em {destino}")

# 📁 Função para mover arquivos específicos
def mover_arquivos(lista, destino_dir):
    for nome in lista:
        if os.path.exists(nome):
            destino_path = os.path.join(destino_dir, nome)
            if not os.path.exists(destino_path):
                shutil.move(nome, destino_path)
                print(f"📁 Arquivo movido: {nome} → {destino_dir}")
            else:
                print(f"⚠️ Já existe: {nome} em {destino_dir}")
        else:
            print(f"❌ Arquivo não encontrado: {nome}")

# 🚀 Execução principal
if __name__ == "__main__":
    criar_pastas(estrutura)

    # Mover imagens por classe
    mover_imagens(".", "imagens/carro", "carro")
    mover_imagens(".", "imagens/moto", "moto")

    # Mover arquivos de modelo
    arquivos_modelo = ["labels.json", "modelo_carro_moto_otimizado.keras"]
    mover_arquivos(arquivos_modelo, "modelo")

    # Mover gráficos
    graficos = ["matriz_confusao.png", "metricas_por_classe.png", "graficos_ft.png"]
    mover_arquivos(graficos, "resultados")

    print("\n🎯 Projeto organizado com sucesso!")

