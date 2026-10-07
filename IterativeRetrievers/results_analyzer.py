import pandas as pd
import math

def fmt(x, width=7):
    if not math.isfinite(x):
        return f"{x:>{width}}"
    for d in range(width - 2, -1, -1):
        s = f"{x:.{d}f}"
        if len(s) <= width:
            return s.rjust(width)
    return s

model_to_price = {
    "deepseek-v4-flash": (0.012, 1.25),
    "kimi-k2.5": (0.45, 2.25),
    "gpt-5.6-luna": (0.02, 1.20),
    "claude-sonnet-5": (2, 10),
    "gpt-4o": (2.5, 10),
    "qwen3.5": (0, 0),
    "gpt-5.4": (2.5, 15)
}
def cost(model_name: str, instances: int, tokens: dict):
    in_cost, out_cost = model_to_price[model_name]
    multiplier = 1_000_000/instances
    results = {}
    for variation in tokens.keys():
        var_in, var_out = tokens[variation]
        curr_cost = var_in/multiplier*in_cost + var_out/multiplier*out_cost
        results[variation] = curr_cost

    return results

CLASSIFIER = "BERT"
DATASET = "2Wiki"
# DATASET = "Hotpot"
# DATASET = "Musique"
# MODEL = "deepseek-v4-flash" # deepseek-v4-flash claude-sonnet-5 deepseek-v4-flash qwen3.8-27b gpt-5.4
MODEL = "qwen3.5" # deepseek-v4-flash claude-sonnet-5 deepseek-v4-flash qwen3.8-27b gpt-5.4
reranker = False
reranker_text = "_reranker" if reranker else ""

df = pd.read_csv(f"../results/{CLASSIFIER}/{MODEL}/vanilla/{DATASET}{reranker_text}.csv")
df = df.dropna(subset=["answer"])

df_suff = pd.read_csv(f"../results/{CLASSIFIER}/{MODEL}/suff/{DATASET}{reranker_text}.csv")
df_suff = df_suff.dropna(subset=["answer"])
df_llm = pd.read_csv(f"../results/{CLASSIFIER}/{MODEL}/llm/{DATASET}{reranker_text}.csv")
df_llm = df_llm.dropna(subset=["answer"])
# df_suff["sufficient"] = np.where(df_suff["sufficient"] == -1, False, True)

df_suff_strat = df_suff[df_suff["sufficient"] != -1]
df_insuff_strat = df_suff[df_suff["sufficient"] == -1]

df_correct = df[df["llm_as_a_judge"] == 1.0]
df_incorrect = df[df["llm_as_a_judge"] != 1.0]
df_suff_correct = df_suff[df_suff["llm_as_a_judge"] == 1.0]
df_suff_incorrect = df_suff[df_suff["llm_as_a_judge"] != 1.0]
df_llm_correct = df_llm[df_llm["llm_as_a_judge"] == 1.0]
df_llm_incorrect = df_llm[df_llm["llm_as_a_judge"] != 1.0]


df_suff_class_correct = df_suff_strat[df_suff_strat["recall@15"] == 1.0]
df_suff_class_incorrect = df_suff_strat[df_suff_strat["recall@15"] != 1.0]
df_insuff_class_correct = df_insuff_strat[df_insuff_strat["recall@15"] != 1.0]
df_insuff_class_incorrect = df_insuff_strat[df_insuff_strat["recall@15"] == 1.0]


# A ideia é ver a taxa de acerto dos que são imediatamente suficientes para saber se, sem o ruído de instâncias excessivas, ele performa melhor.
df_suff_strat_immediately = df_suff[df_suff["sufficient"] == 1]
df_insuff_strat_immediately = df_suff[df_suff["sufficient"] != 1]

df_suff_class_correct_immediately = df_suff_strat_immediately[(df_suff_strat_immediately["recall@5"] == 1.0)]
df_suff_class_incorrect_immediately = df_suff_strat_immediately[(df_suff_strat_immediately["recall@5"] != 1.0)]
df_insuff_class_correct_immediately = df_insuff_strat_immediately[df_insuff_strat_immediately["recall@5"] != 1.0]
df_insuff_class_incorrect_immediately = df_insuff_strat_immediately[df_insuff_strat_immediately["recall@5"] == 1.0]

#====================================PRINTING====================================
print(f"Analysis of {MODEL}'s performance on {DATASET}, with{"out" if not reranker else ""} reranker")
print("="*20)
print("Performance:")

output_llm_rate = df_llm["output_tokens_generated"].sum() / (df_llm["input_tokens_used"].sum() + df_llm["output_tokens_generated"].sum()) # Estimates that the model outputs around the same proportion of tokens as it did for the answer. In fact, it is probably much more, but I think this is more fair
usage = {
    'std': (df["input_tokens_used"].mean(), df["output_tokens_generated"].mean()),
    'suff': (df_suff["input_tokens_used"].mean(), df_suff["output_tokens_generated"].mean()),
    'llm': (df_llm["input_tokens_used"].mean() + df_llm["sufficiency_tokens_used"].mean()*(1-output_llm_rate), df_llm["output_tokens_generated"].mean() + df_llm["sufficiency_tokens_used"].mean()*output_llm_rate),
}
costs = cost(MODEL, df.shape[0], usage)
print("       | time  | intok |gen tok|sufftok| turns |cost($)|")
print(f"  std  |{fmt(df["inference_time"].mean())}|{fmt(df["input_tokens_used"].mean())}|{fmt(df["output_tokens_generated"].mean())}|   -   |{fmt(df["turns"].mean())}|{fmt(costs["std"])}|")
print(f" suff  |{fmt(df_suff["inference_time"].mean())}|{fmt(df_suff["input_tokens_used"].mean())}|{fmt(df_suff["output_tokens_generated"].mean())}|{fmt(df_suff["sufficiency_tokens_used"].mean())}|{fmt(df_suff["turns"].mean())}|{fmt(costs["suff"])}|") 
print(f"  llm  |{fmt(df_llm["inference_time"].mean())}|{fmt(df_llm["input_tokens_used"].mean())}|{fmt(df_llm["output_tokens_generated"].mean())}|{fmt(df_llm["sufficiency_tokens_used"].mean())}|{fmt(df_llm["turns"].mean())}|{fmt(costs["llm"])}|") 
print("="*20)
print("Desempenho nas respostas:")
print("       |  EM   |  F1   |  LLM  |ROUGE-L|")
print(f"  std  |{fmt(df["exact_match"].mean())}|{fmt(df["f1"].mean())}|{fmt(df["llm_as_a_judge"].mean())}|{fmt(df["rouge"].mean())}|")
print(f" suff  |{fmt(df_suff["exact_match"].mean())}|{fmt(df_suff["f1"].mean())}|{fmt(df_suff["llm_as_a_judge"].mean())}|{fmt(df_suff["rouge"].mean())}|") 
print(f"  llm  |{fmt(df_llm["exact_match"].mean())}|{fmt(df_llm["f1"].mean())}|{fmt(df_llm["llm_as_a_judge"].mean())}|{fmt(df_llm["rouge"].mean())}|") 
print(f"Total exact matches: std = {df["exact_match"].sum()} | suff = {df_suff["exact_match"].sum()} | llm = {df_llm["exact_match"].sum()}")
print(f"Total LLM-as-a-judge correctness: std = {df["llm_as_a_judge"].sum()} | suff = {df_suff["llm_as_a_judge"].sum()} | llm = {df_llm["llm_as_a_judge"].sum()}")
print("="*20)
print("Desempenho no retrieval:")
print("       |  MAP  | MAP@5 |  R@5  |  P@5  | R@15  | P@15  |PR_AUC |")
print(f"  std  |{fmt(df['average_precision'].mean())}|{fmt(df['average_precision@5'].mean())}|{fmt(df['recall@5'].mean())}|{fmt(df['precision@5'].mean())}|{fmt(df['recall@15'].mean())}|{fmt(df['precision@15'].mean())}|{fmt(df['pr_auc'].mean())}|")
print(f" suff  |{fmt(df_suff['average_precision'].mean())}|{fmt(df_suff['average_precision@5'].mean())}|{fmt(df_suff['recall@5'].mean())}|{fmt(df_suff['precision@5'].mean())}|{fmt(df_suff['recall@15'].mean())}|{fmt(df_suff['precision@15'].mean())}|{fmt(df_suff['pr_auc'].mean())}|")
print(f"  llm  |{fmt(df_llm['average_precision'].mean())}|{fmt(df_llm['average_precision@5'].mean())}|{fmt(df_llm['recall@5'].mean())}|{fmt(df_llm['precision@5'].mean())}|{fmt(df_llm['recall@15'].mean())}|{fmt(df_llm['precision@15'].mean())}|{fmt(df_llm['pr_auc'].mean())}|")
print("="*20)
print("Desempenho no retrieval de acordo com correteza:")
print("       |  hit  |  MAP  | MAP@5 |  R@5  |  P@5  | R@15  | P@15  |PR_AUC |")
print(f"  std  |  1.0  |{fmt(df_correct['average_precision'].mean())}|{fmt(df_correct['average_precision@5'].mean())}|{fmt(df_correct['recall@5'].mean())}|{fmt(df_correct['precision@5'].mean())}|{fmt(df_correct['recall@15'].mean())}|{fmt(df_correct['precision@15'].mean())}|{fmt(df_correct['pr_auc'].mean())}|")
print(f"  std  |  0.0  |{fmt(df_incorrect['average_precision'].mean())}|{fmt(df_incorrect['average_precision@5'].mean())}|{fmt(df_incorrect['recall@5'].mean())}|{fmt(df_incorrect['precision@5'].mean())}|{fmt(df_incorrect['recall@15'].mean())}|{fmt(df_incorrect['precision@15'].mean())}|{fmt(df_incorrect['pr_auc'].mean())}|")
print(f" suff  |  1.0  |{fmt(df_suff_correct['average_precision'].mean())}|{fmt(df_suff_correct['average_precision@5'].mean())}|{fmt(df_suff_correct['recall@5'].mean())}|{fmt(df_suff_correct['precision@5'].mean())}|{fmt(df_suff_correct['recall@15'].mean())}|{fmt(df_suff_correct['precision@15'].mean())}|{fmt(df_suff_correct['pr_auc'].mean())}|")
print(f" suff  |  0.0  |{fmt(df_suff_incorrect['average_precision'].mean())}|{fmt(df_suff_incorrect['average_precision@5'].mean())}|{fmt(df_suff_incorrect['recall@5'].mean())}|{fmt(df_suff_incorrect['precision@5'].mean())}|{fmt(df_suff_incorrect['recall@15'].mean())}|{fmt(df_suff_incorrect['precision@15'].mean())}|{fmt(df_suff_incorrect['pr_auc'].mean())}|")
print(f"  llm  |  1.0  |{fmt(df_llm_correct['average_precision'].mean())}|{fmt(df_llm_correct['average_precision@5'].mean())}|{fmt(df_llm_correct['recall@5'].mean())}|{fmt(df_llm_correct['precision@5'].mean())}|{fmt(df_llm_correct['recall@15'].mean())}|{fmt(df_llm_correct['precision@15'].mean())}|{fmt(df_llm_correct['pr_auc'].mean())}|")
print(f"  llm  |  0.0  |{fmt(df_llm_incorrect['average_precision'].mean())}|{fmt(df_llm_incorrect['average_precision@5'].mean())}|{fmt(df_llm_incorrect['recall@5'].mean())}|{fmt(df_llm_incorrect['precision@5'].mean())}|{fmt(df_llm_incorrect['recall@15'].mean())}|{fmt(df_llm_incorrect['precision@15'].mean())}|{fmt(df_llm_incorrect['pr_auc'].mean())}|")
print("="*20)
print(f"Erros do LLM as a Judge: std = {df[(df["exact_match"] == 1.0) & (df["llm_as_a_judge"] == 0)].shape[0]} | suff = {df_suff[(df_suff["exact_match"] == 1.0) & (df_suff["llm_as_a_judge"] == 0)].shape[0]} | llm = {df_llm[(df_llm["exact_match"] == 1.0) & (df_llm["llm_as_a_judge"] == 0)].shape[0]}")
print(f"Correções do LLM as a Judge: std = {df[(df["exact_match"] == 0.0) & (df["llm_as_a_judge"] == 1.0)].shape[0]} | suff = {df_suff[(df_suff["exact_match"] == 0.0) & (df_suff["llm_as_a_judge"] == 1.0)].shape[0]} | llm = {df_llm[(df_llm["exact_match"] == 0.0) & (df_llm["llm_as_a_judge"] == 1.0)].shape[0]}")

print("~~~"*20)

print("Importância da suficiência (Médias estratificadas pelo encontro da suficiência no modelo BERT):")
print("Desempenho nas respostas:")
print(f"Sufficient instances: {df_suff_strat.shape[0]} | Insufficient instances: {df_insuff_strat.shape[0]}")
print("       |  EM   |  F1   |  LLM  |ROUGE-L|")
print(f" suff  |{fmt(df_suff_strat["exact_match"].mean())}|{fmt(df_suff_strat["f1"].mean())}|{fmt(df_suff_strat["llm_as_a_judge"].mean())}|{fmt(df_suff_strat["rouge"].mean())}|")
print(f"insuff |{fmt(df_insuff_strat["exact_match"].mean())}|{fmt(df_insuff_strat["f1"].mean())}|{fmt(df_insuff_strat["llm_as_a_judge"].mean())}|{fmt(df_insuff_strat["rouge"].mean())}|") 
print(f"Total exact matches: suff = {df_suff_strat["exact_match"].sum()} | insuff = {df_insuff_strat["exact_match"].sum()}")
print(f"Total LLM-as-a-judge correctness: suff = {df_suff_strat["llm_as_a_judge"].sum()} | insuff = {df_insuff_strat["llm_as_a_judge"].sum()}")
print("="*20)
print("Desempenho no retrieval:")
print("       |  MAP  | MAP@5 |  R@5  |  P@5  | R@15  | P@15  |PR_AUC |")
print(f" suff  |{fmt(df_suff_strat['average_precision'].mean())}|{fmt(df_suff_strat['average_precision@5'].mean())}|{fmt(df_suff_strat['recall@5'].mean())}|{fmt(df_suff_strat['precision@5'].mean())}|{fmt(df_suff_strat['recall@15'].mean())}|{fmt(df_suff_strat['precision@15'].mean())}|{fmt(df_suff_strat['pr_auc'].mean())}|")
print(f"insuff |{fmt(df_insuff_strat['average_precision'].mean())}|{fmt(df_insuff_strat['average_precision@5'].mean())}|{fmt(df_insuff_strat['recall@5'].mean())}|{fmt(df_insuff_strat['precision@5'].mean())}|{fmt(df_insuff_strat['recall@15'].mean())}|{fmt(df_insuff_strat['precision@15'].mean())}|{fmt(df_insuff_strat['pr_auc'].mean())}|")
print("="*20)
print("Acerto da suficiência:")
print("Matriz de confusão:")
print("T\\pred |insuff | suff  |")
print(f"insuff |{df_insuff_class_correct.shape[0]:7}|{df_suff_class_incorrect.shape[0]:7}|")
print(f" suff  |{df_insuff_class_incorrect.shape[0]:7}|{df_suff_class_correct.shape[0]:7}|")
# print(f"Instâncias acertadas: suff = {df_suff_class_correct.shape[0]} | insuff = {df_insuff_class_correct.shape[0]}")
print(f"Acurácia do classificador de suficência: {(df_insuff_class_correct.shape[0] + df_suff_class_correct.shape[0])/df_suff.shape[0]} ")
prec = df_suff_class_correct.shape[0] / (df_suff_class_correct.shape[0] + df_suff_class_incorrect.shape[0])
recall = df_suff_class_correct.shape[0] / (df_suff_class_correct.shape[0] + df_insuff_class_incorrect.shape[0])
print(f"Precisão do classificador de suficência: {prec} ")
print(f"Recall do classificador de suficência: {recall} ")
print(f"F1 do classificador de suficência: {(2*prec*recall)/(prec+recall)} ")
print("="*20)
print("Matriz de confusão @5:")
print("T\\pred |insuff | suff  |")
print(f"insuff |{df_insuff_class_correct_immediately.shape[0]:7}|{df_suff_class_incorrect_immediately.shape[0]:7}|")
print(f" suff  |{df_insuff_class_incorrect_immediately.shape[0]:7}|{df_suff_class_correct_immediately.shape[0]:7}|")
# print(f"Instâncias acertadas: suff = {df_suff_class_correct_immediately.shape[0]} | insuff = {df_insuff_class_correct_immediately.shape[0]}")
print(f"Acurácia do classificador de suficência: {(df_insuff_class_correct_immediately.shape[0] + df_suff_class_correct_immediately.shape[0])/df_suff.shape[0]} ")
prec = df_suff_class_correct_immediately.shape[0] / (df_suff_class_correct_immediately.shape[0] + df_suff_class_incorrect_immediately.shape[0])
recall = df_suff_class_correct_immediately.shape[0] / (df_suff_class_correct_immediately.shape[0] + df_insuff_class_incorrect_immediately.shape[0])
print(f"Precisão do classificador de suficência: {prec} ")
print(f"Recall do classificador de suficência: {recall} ")
print(f"F1 do classificador de suficência: {(2*prec*recall)/(prec+recall)} ")
# print(" F1  |") sufficient,,average_precision,recall@5,precision@5,pr_auc

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Expandir o DataFrame criando uma linha para cada turno (1 ate 'turns')
df_expanded = df_suff.loc[df_suff.index.repeat(df_suff["turns"])].copy()
df_expanded["turn"] = df_expanded.groupby(level=0).cumcount() + 1

# 2. Definir o Valor Real (Ground Truth) para cada turno
# O contexto e suficiente no turno t se turn_recall_1 != -1 e t >= turn_recall_1
y_true = np.where(
    (df_expanded["turn_recall_1"] != -1)
    & (df_expanded["turn"] >= df_expanded["turn_recall_1"]), # Porque só pode classificar como suficiente depois que o recall virou 1
    1,
    0,
)

# 3. Definir o Valor Predito (Modelo) para cada turno
# O modelo considerou suficiente no turno t se sufficient != -1 e t >= sufficient
y_pred = np.where(
    (df_expanded["sufficient"] != -1)
    & (df_expanded["turn"] >= df_expanded["sufficient"]), # Porque nos turnos anteriores a ele classificar como sufficient ele classificou como insuficiente
    1,
    0,
)

# 4. Matriz de Confusão
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Exibir os resultados organizados
cm_df = pd.DataFrame(
    cm,
    index=["Real: Insuficiente (0)", "Real: Suficiente (1)"],
    columns=["Pred: Insuficiente (0)", "Pred: Suficiente (1)"],
)

print("Matriz de Confusão:")
print(cm_df)

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=["Insuficiente", "Suficiente"]))

print("~~~"*20)

df_suff_strat = df_llm[df_llm["sufficient"] != -1]
df_insuff_strat = df_llm[df_llm["sufficient"] == -1]

df_correct = df[df["llm_as_a_judge"] == 1.0]
df_incorrect = df[df["llm_as_a_judge"] != 1.0]
df_suff_correct = df_suff[df_suff["llm_as_a_judge"] == 1.0]
df_suff_incorrect = df_suff[df_suff["llm_as_a_judge"] != 1.0]


df_suff_class_correct = df_suff_strat[df_suff_strat["recall@15"] == 1.0]
df_suff_class_incorrect = df_suff_strat[df_suff_strat["recall@15"] != 1.0]
df_insuff_class_correct = df_insuff_strat[df_insuff_strat["recall@15"] != 1.0]
df_insuff_class_incorrect = df_insuff_strat[df_insuff_strat["recall@15"] == 1.0]

print("Importância da suficiência (Médias estratificadas pelo encontro da suficiência com a LLM):")
print("Desempenho nas respostas:")
print(f"Sufficient instances: {df_suff_strat.shape[0]} | Insufficient instances: {df_insuff_strat.shape[0]}")
print("       |  EM   |  F1   |  LLM  |ROUGE-L|")
print(f" suff  |{fmt(df_suff_strat["exact_match"].mean())}|{fmt(df_suff_strat["f1"].mean())}|{fmt(df_suff_strat["llm_as_a_judge"].mean())}|{fmt(df_suff_strat["rouge"].mean())}|")
print(f"insuff |{fmt(df_insuff_strat["exact_match"].mean())}|{fmt(df_insuff_strat["f1"].mean())}|{fmt(df_insuff_strat["llm_as_a_judge"].mean())}|{fmt(df_insuff_strat["rouge"].mean())}|") 
print(f"Total exact matches: suff = {df_suff_strat["exact_match"].sum()} | insuff = {df_insuff_strat["exact_match"].sum()}")
print(f"Total LLM-as-a-judge correctness: suff = {df_suff_strat["llm_as_a_judge"].sum()} | insuff = {df_insuff_strat["llm_as_a_judge"].sum()}")
print("="*20)
print("Desempenho no retrieval:")
print("       |  MAP  | MAP@5 |  R@5  |  P@5  | R@15  | P@15  |PR_AUC |")
print(f" suff  |{fmt(df_suff_strat['average_precision'].mean())}|{fmt(df_suff_strat['average_precision@5'].mean())}|{fmt(df_suff_strat['recall@5'].mean())}|{fmt(df_suff_strat['precision@5'].mean())}|{fmt(df_suff_strat['recall@15'].mean())}|{fmt(df_suff_strat['precision@15'].mean())}|{fmt(df_suff_strat['pr_auc'].mean())}|")
print(f"insuff |{fmt(df_insuff_strat['average_precision'].mean())}|{fmt(df_insuff_strat['average_precision@5'].mean())}|{fmt(df_insuff_strat['recall@5'].mean())}|{fmt(df_insuff_strat['precision@5'].mean())}|{fmt(df_insuff_strat['recall@15'].mean())}|{fmt(df_insuff_strat['precision@15'].mean())}|{fmt(df_insuff_strat['pr_auc'].mean())}|")
print("="*20)
print("Acerto da suficiência:")
print("Matriz de confusão:")
print(" T\\pred |insuff | suff  |")
print(f"insuff |{df_insuff_class_correct.shape[0]:7}|{df_suff_class_incorrect.shape[0]:7}|")
print(f" suff  |{df_insuff_class_incorrect.shape[0]:7}|{df_suff_class_correct.shape[0]:7}|")
# print(f"Instâncias acertadas: suff = {df_suff_class_correct.shape[0]} | insuff = {df_insuff_class_correct.shape[0]}")
print(f"Acurácia do classificador de suficência: {(df_insuff_class_correct.shape[0] + df_suff_class_correct.shape[0])/df_suff.shape[0]} ")
prec = df_suff_class_correct.shape[0] / (df_suff_class_correct.shape[0] + df_suff_class_incorrect.shape[0])
recall = df_suff_class_correct.shape[0] / (df_suff_class_correct.shape[0] + df_insuff_class_incorrect.shape[0])
print(f"Precisão do classificador de suficência: {prec} ")
print(f"Recall do classificador de suficência: {recall} ")
print(f"F1 do classificador de suficência: {(2*prec*recall)/(prec+recall)} ")
print("="*20)

# 1. Expandir o DataFrame criando uma linha para cada turno (1 ate 'turns')
df_expanded = df_llm.loc[df_llm.index.repeat(df_llm["turns"])].copy()
df_expanded["turn"] = df_expanded.groupby(level=0).cumcount() + 1

# 2. Definir o Valor Real (Ground Truth) para cada turno
# O contexto e suficiente no turno t se turn_recall_1 != -1 e t >= turn_recall_1
y_true = np.where(
    (df_expanded["turn_recall_1"] != -1)
    & (df_expanded["turn"] >= df_expanded["turn_recall_1"]), # Porque só pode classificar como suficiente depois que o recall virou 1
    1,
    0,
)

# 3. Definir o Valor Predito (Modelo) para cada turno
# O modelo considerou suficiente no turno t se sufficient != -1 e t >= sufficient
y_pred = np.where(
    (df_expanded["sufficient"] != -1)
    & (df_expanded["turn"] >= df_expanded["sufficient"]), # Porque nos turnos anteriores a ele classificar como sufficient ele classificou como insuficiente
    1,
    0,
)

# 4. Matriz de Confusão
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Exibir os resultados organizados
cm_df = pd.DataFrame(
    cm,
    index=["Real: Insuficiente (0)", "Real: Suficiente (1)"],
    columns=["Pred: Insuficiente (0)", "Pred: Suficiente (1)"],
)

print("Matriz de Confusão:")
print(cm_df)

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=["Insuficiente", "Suficiente"]))

print("="*20)
print("Concordância entre modelo e LLM-as-a-Judge")
suff_suff = df_suff[(df_suff["sufficient"] != -1) & (df_llm["sufficient"] != -1)].shape[0]
suff_insuff = df_suff[(df_suff["sufficient"] != -1) & (df_llm["sufficient"] == -1)].shape[0]
insuff_suff = df_suff[(df_suff["sufficient"] == -1) & (df_llm["sufficient"] != -1)].shape[0]
insuff_insuff = df_suff[(df_suff["sufficient"] == -1) & (df_llm["sufficient"] == -1)].shape[0]
print("Matriz de concordância entre LLM :")
print("bert\\llm| insuff |  suff  |")
print(f" insuff |{suff_insuff:8}|{insuff_suff:8}|")
print(f"  suff  |{insuff_insuff:8}|{suff_suff:8}|")