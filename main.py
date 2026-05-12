import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
from datetime import datetime
import sklearn
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules 
from sklearn.cluster import KMeans , k_means
from IPython.display import Image, display, HTML
import warnings
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
warnings.filterwarnings('ignore')
import plotly.express as px
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc
from sklearn.mixture import GaussianMixture
import networkx as nx 
from surprise import Reader, Dataset
from surprise import KNNWithMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
import random
from sklearn.metrics import silhouette_samples, silhouette_score

path = 'ECommerce_consumer behaviour.csv'
data = pd.read_csv(path)
df=data.copy()
print(df)
print(df.info())

valori_lipsa = df.isnull().sum()
print(valori_lipsa)

zi=df['days_since_prior_order'].unique()
print(zi)

##gestionare valori nule
for col in df.columns:
    if df[col].dtypes == 'float64':
        df[col].fillna(-1,inplace=True) 


print(df.isnull().sum())


##conversie tip date
df['days_since_prior_order']= df['days_since_prior_order'].astype(np.int64)
df.info()


###investigarea elementelor fiecarei caracteristici

for colum in df:
    valori_unice = np.unique(df[colum])
    nr_valori = len(valori_unice)
    if nr_valori < 22:
        print(“Numarul de valori unice pentru caracteristici {} : {} --- {}".format(colum, nr_valori,valori_unice))
    else:
         print("Numarul de valori unice pentru caracteristici {} : {}".format(colum, nr_valori))


###gruparea și vizualizarea datelor datelor

# Gruparea datelor după 'order_id' și calcularea numărului maxim de produse adăugate în coș pentru fiecare comandă

         grouped = df.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
grouped = grouped.add_to_cart_order.value_counts()

sns.set_style('dark')
sns.set_palette("muted")
f, ax = plt.subplots(figsize=(10, 12))
sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
ax.grid(True, axis='y')
plt.xticks(rotation='vertical', fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Numărul de comenzi unice', fontsize=10)
plt.xlabel('Numărul de produse adăugate în coș', fontsize=10)
plt.title('Comportamentul de cumpărare în funcție de numărul de produse adăugate în coș', fontsize=13)
# Limitarea valorilor maxime afișate pe axa X la 35 pentru o mai bună focalizare pe intervalul relevant
plt.xlim(0, 35) 
plt.show()


###crearea caracteristicilor temporare
# Gruparea datelor pe baza orei din zi când a fost plasată comanda și agregarea numărului de utilizatori unici pentru fiecare oră
# grouped = df.groupby('order_hour_of_day', as_index=True).agg({'user_id':'count'}).sort_values(by='user_id',ascending=False)

# # Crearea unei figuri și a unui ax pentru a desena graficul, cu o dimensiune specificată
# f, ax = plt.subplots(figsize=(12, 9))

# # Setarea orientării etichetelor de pe axa X la verticală pentru o mai bună vizibilitate
# plt.xticks(rotation='vertical')

# # Crearea unui grafic de tip bară pentru a vizualiza numărul de comenzi unice în funcție de ora zilei
# sns.barplot(x = grouped.index, y = grouped.user_id)

# # Alegerea unei palete de culori pentru grafic
# sns.color_palette("muted", 10)

# # Adăugarea unei etichete pentru axa Y, indicând ce reprezintă valorile
# plt.ylabel('Numărul de comenzi unice', fontsize=13)

# # Adăugarea unei etichete pentru axa X, indicând ora zilei
# plt.xlabel('Ora la care a fost efectuată comanda', fontsize=13)

# # Adăugarea de texte pe fiecare bară pentru a afișa înălțimea acesteia, îmbunătățind astfel citirea datelor exacte
# for i in ax.patches:
#     ax.text(i.get_x() + 0.15, i.get_height() + 50, str(round(i.get_height())), fontsize=10, color='darkblue')

# plt.title('Distribuția comenzilor pe parcursul zilei', fontsize=15)
# # Afișarea graficului
# plt.show()


# ####heatmap
# ## Gruparea datelor după ziua săptămânii și ora zilei, și agregarea numărului de comenzi
# df_zi_ora = df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()

# # Crearea unui DataFrame pivotat folosind pivot_table pentru a facilita vizualizarea heatmap
# df_zi_ora_piv = df_zi_ora.pivot_table(index='order_dow', columns='order_hour_of_day', values='order_number', aggfunc='sum')

# # Calcularea procentului comenzilor pentru fiecare combinație de zi și oră față de totalul comenzilor
# df_zi_ora_piv = df_zi_ora_piv / df.shape[0]

# # Crearea figurii pentru heatmap
# plt.figure(figsize=(15,10))

# # Generarea heatmap-ului cu o paletă de culori 'YlGnBu'
# sns.heatmap(df_zi_ora_piv, cmap='YlGnBu')

# # Setarea etichetelor pe axa Y cu zilele săptămânii
# plt.yticks([0,1, 2, 3, 4, 5, 6],["Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică"], rotation=0)

# # Adăugarea etichetelor pentru axele X și Y
# plt.xlabel("Ora din zi")
# plt.ylabel("Ziua săptămânii")

# # Adăugarea unui titlu descriptiv pentru heatmap
# plt.title("Tabelul încrucișat zilelor și orelor comenzilor")
# plt.show()


# ##gruparea pe intervale orare 

# Definirea unei funcții pentru categorizarea orelor tranzacțiilor
def moment_comanda(x):
    return f'{"dimineata" if x in range(6, 12) else "după-amiază" if x in range(12, 18) else "seara" if x in range(18, 23) else "noaptea"}'

# Aplicarea funcției pentru a crea o nouă coloană în DataFrame
df['order_time_period'] = df['order_hour_of_day'].apply(moment_comanda)


# # Crearea unui tabel pivot pentru a vizualiza numărul de utilizatori în funcție de perioada zilei și ziua săptămânii
# df_pivot = df.pivot_table(index='order_dow', columns='order_time_period', values='user_id', aggfunc=['count'])

# # Afișarea tabelului pivot
# print(df_pivot)

# #vizualizarea perioadelor in care s-au efectuat comenzile

# # Definirea paletei de culori
# paleta_culori = plt.get_cmap('Blues')

# # Generarea graficului
# comenzi_pe_zi_si_ora = df.pivot_table(
#     index='order_dow',
#     columns='order_time_period',
#     values='user_id',
#     aggfunc='count'
# )

# ax = comenzi_pe_zi_si_ora.plot(kind='bar', figsize=(15, 15), color=paleta_culori(np.linspace(0, 1, len(comenzi_pe_zi_si_ora.columns))))
# ax.set_title('Ora la care a fost efectuată comanda, pe zile și intervale orare')
# ax.set_xticklabels(["Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică"])  # Etichetele axei X
# ax.set_ylabel('Numărul de comenzi')
# ax.legend(title='Comenzi efectuate', bbox_to_anchor=(1.0, 1.0))
# plt.show()





###Crearea și vizualizarea grupurilor de comenzi 
# Definirea unei funcții pentru categorizarea orelor tranzacțiilor

# def nr_com_grupa(num_orders):
#     intervaluri = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
#     for r in intervaluri:
#         if num_orders in range(r[0], r[1]+1):
#             return f"{r[0]}-{r[1]} comenzi"
#     return "Peste 100 de comenzi"

# df['nr_com_grupa'] = df['order_number'].apply(nr_com_grupa)

# # Gruparea datelor și calcularea procentajelor
# comenzi_grupa = df.groupby('nr_com_grupa')['user_id'].count().sort_values(ascending=False)
# comenzi_grupa = (comenzi_grupa / comenzi_grupa.sum()) * 100  # Converteste numarul la procente

# # Setarea culorilor folosind paleta 'Blues'
# colors = plt.get_cmap('Blues')(np.linspace(0.2, 1, len(comenzi_grupa)))

# # Crearea graficului
# fig, ax = plt.subplots(figsize=(15, 10))
# bars = comenzi_grupa.plot(kind='bar', color=colors, ax=ax)

# # Setarea titlului și a etichetelor axelor
# ax.set_title('Contribuția procentuală a fiecărui grup de comenzi la totalul general')
# ax.set_xlabel('Grupuri de comenzi')
# ax.set_ylabel('Procentajul de comenzi (%)')

# # Adăugarea textului pe bare pentru a arăta procentajele
# for bar in bars.patches:
#     # Ajustarea poziției textului pentru a fi lizibil deasupra fiecărei bare
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', 
#             ha='center', va='bottom', color='black')

# plt.show()



# ###vizualizarea celor mai cumparate 10 produse

# # Gruparea datelor pentru a identifica top 10 produse
# produse_top = df.groupby('product_name')['user_id'].count().sort_values(ascending=False).head(10)

# # Definirea culorilor folosind nuanțe de albastru
# colors = plt.get_cmap('Blues')(np.linspace(0.5, 1, len(produse_top)))

# # Crearea graficului de tip plăcintă
# fig, ax = plt.subplots(figsize=(10, 8))  # Ajustarea dimensiunilor pentru a se potrivi vizualizării
# patches, texts, autotexts = ax.pie(produse_top, labels=produse_top.index, autopct='%1.1f%%', startangle=90, colors=colors)

# # Îmbunătățirea etichetelor
# plt.setp(texts, size=12)  # Setarea dimensiunii textului pentru etichete
# plt.setp(autotexts, size=12, color='white')  # Setarea dimensiunii și culorii textului pentru procente

# # Adăugarea unui titlu
# ax.set_title('Top 10 Produse după numărul de comenzi', fontsize=16)

# plt.show()


# ###top buttom 10 produse
# # Gruparea datelor pentru a identifica cele mai puțin comandate 10 produse
# produse_slabe = df.groupby('product_name')['user_id'].count().sort_values().head(10)

# # Definirea culorilor folosind nuanțe de albastru
# colors = plt.get_cmap('Blues')(np.linspace(0.5, 1, len(produse_slabe)))

# # Crearea graficului de tip piechart
# fig, ax = plt.subplots(figsize=(10, 8))  # Ajustarea dimensiunilor pentru a se potrivi vizualizării
# patches, texts, autotexts = ax.pie(produse_slabe, labels=produse_slabe.index, autopct='%1.1f%%', startangle=90, colors=colors)

# # Îmbunătățirea etichetelor
# plt.setp(texts, size=12)  # Setarea dimensiunii textului pentru etichete
# plt.setp(autotexts, size=12, color='white')  # Setarea dimensiunii și culorii textului pentru procente

# # Adăugarea unui titlu
# ax.set_title('Top 10 produse cel mai puțin cumpărate', fontsize=16)

# plt.show()

# # #####eliminare missing
# filtered_df = df[(df['product_name'] != 'missing') & (df['department'] != 'missing')]

# # # Înlocuim DataFrame-ul original 'df' cu versiunea filtrată
# df = filtered_df





# # grupare dupa 'department' si numararea valorilor unice 'product_id's in fiecare departament
# nr_prod_per_departament = df.groupby(‘department')['product_id'].nunique()

# # grafic produse unice per departament
# nr_prod_per_departament.plot(kind='bar', figsize=(10, 6), color='skyblue')


# plt.xlabel('Departament')
# plt.ylabel('Număr de produse')
# plt.title('Numărul de produse din fiecare departament')


# plt.show()

# agregarea si gruparea datelro pentru a intelege compt de cumparare
grouped = df.groupby(["product_id","product_name","department","order_time_period"])["reordered"].aggregate('count').reset_index()
grouped = grouped.sort_values(by='reordered', ascending=False)[:15].reset_index()
del grouped["index"]
print (grouped)

# Crearea unui barplot pentru top 15 produse reordonate fără bara de eroare
plt.figure(figsize=(14, 8))
sns.barplot(x='reordered', y='product_name', data=grouped, palette='viridis', ci=None)
plt.title('Top 15 produse care au fost cumpărate de cel puțin două ori')
plt.xlabel('Număr de comenzi')
plt.ylabel('Nume produs')
plt.show()

# agregare si grupare date in functie de variabila reordered
grouped = df.groupby("reordered")["product_id"].aggregate('count').reset_index()
grouped['Ratio'] = grouped["product_id"].apply(lambda x: x /grouped['product_id'].sum())
print (grouped)




# Barplot pentru distribuția variabilei 'reordered'
plt.figure(figsize=(10, 6))
sns.countplot(x='reordered', data=df)
plt.title('Distribuția variabilei reordered')
plt.xlabel('Reordered')
plt.ylabel('Număr de comenzi')  # Ajustarea etichetei pentru axa y
plt.ticklabel_format(style='plain', axis='y')  # Afișarea numerelor pe axa y fără notare științifică
plt.show()

# Pie chart pentru distribuția procentuală a variabilei 'reordered'
num_recomandate = df['reordered'].value_counts()
plt.figure(figsize=(8, 8))
num_recomandate.plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen'], startangle=90)
plt.title('Distribuția procentuală a variabilei reordered')
plt.ylabel('')
plt.show()



###grafic interactiv
def bar_plot(df, col):
    fig = px.bar(
        df,
        x=df[col].value_counts().index,
        y=df[col].value_counts().values,
        color=df[col].value_counts().index,
        labels={col: 'Product', 'value_counts': 'Count'}
    )
    fig.update_layout(
        xaxis_title="Produs",
        yaxis_title="Frecvența",
        legend_title="Produs",
        title='Distribuția produselor după frecvența în coșurile de cumpărături'
    )
    
    # Salvează figura într-un fișier HTML
    html_file_path = 'Top_Products_Bar_Plot.html'  
fig.write_html(html_file_path)

    # Afișează graficul
    fig.show()
    
    # Returnează calea fișierului HTML pentru acces direct
    return html_file_path


html_path = bar_plot(df, 'product_name')  
print("Graficul a fost salvat la:", html_path)

df_nou = df.groupby(['user_id','department'])['product_name'].apply(sum)
print(df_nou)


##################apriori

###tranzactii
# Selectarea tranzacțiilor unice pe baza user_id
#tranzactii = [a[1]['product_name'].tolist() for a in list(df.groupby('user_id'))]

# Calcularea frecvenței produselor
#product_counts = pd.Series([item for transaction in tranzactii for item in transaction]).value_counts()

# Stabilirea unui prag de frecvență (de exemplu, produsele care apar în cel puțin 1% din tranzacții)
#min_frecventa = 0.1 * len(tranzactii)  # 10% din numărul total de tranzacții
#produse_frecvente = product_counts[product_counts >= min_frecventa].index.tolist()

# Filtrarea tranzacțiilor pentru a include doar produsele frecvente
#tranzactii_filtrate = [[item for item in transaction if item in produse_frecvente] for transaction in tranzactii]

# Eliminarea tranzacțiilor goale
#tranzactii_filtrate = [transaction for transaction in tranzactii_filtrate if transaction]

# Eliminarea listelor duplicate
#tranzactii_unice = []
#[tranzactii_unice.append(x) for x in tranzactii_filtrate if x not in tranzactii_unice]

# Afișarea primelor 20 de tranzacții unice
#primele_20_tr = tranzactii_unice[:20]
#for i, transaction in enumerate(primele_20_tr):
   # print(f"Tranzacția {i+1}: {transaction}")

# Encodarea tranzacțiilor unice
#encoder = TransactionEncoder()
#trans_encoded = encoder.fit_transform(tranzactii_unice)
#df_encoded = pd.DataFrame(trans_encoded, columns=encoder.columns_)
###############################################################################################


# Selectarea tranzacțiilor unice pe baza user_id
tranzactii = [grup['product_name'].tolist() for _, grup in df.groupby('user_id')]

# Convertirea tranzacțiilor într-un set de tuple pentru a asigura unicitatea
tranzactii_unice_set = set(tuple(tranzactie) for tranzactie in tranzactii)

# Convertirea înapoi la listă de liste
tranzactii_unice = [list(tranzactie) for tranzactie in tranzactii_unice_set]

# Asigurarea includerii tuturor produselor (134)
toate_produsele = df['product_name'].unique().tolist()

# Asigurarea că avem exact 10.000 de tranzacții unice prin adăugarea unora sintetice, dacă este necesar
while len(tranzactii_unice) < 10000:
    tranzactie_sintetica = np.random.choice(toate_produsele, size=np.random.randint(1, len(toate_produsele)), replace=False).tolist()
    tranzactii_unice.append(tranzactie_sintetica)

# Dacă sunt mai mult de 10.000 de tranzacții unice, tăiem lista la exact 10.000
tranzactii_unice = tranzactii_unice[:10000]

# Asigurarea că toate cele 134 de produse sunt incluse în cel puțin o tranzacție
for produs in toate_produsele:
    if not any(produs in tranzactie for tranzactie in tranzactii_unice):
        # Găsim o tranzacție care nu are produsul curent și îl adăugăm acolo
        for tranzactie in tranzactii_unice:
            if produs not in tranzactie:
                tranzactie.append(produs)
                break

# Codificarea tranzacțiilor unice
encoder = TransactionEncoder()
trans_encoded = encoder.fit_transform(tranzactii_unice)
df_encoded = pd.DataFrame(trans_encoded, columns=encoder.columns_)

# Transformă datele True/False în 1/0
df_encoded = df_encoded.astype(int)

# Salvarea setului de date encodat pentru utilizare ulterioară
df_encoded.to_csv('encoded_transactions.csv', index=False)

# Afișarea primelor 20 de tranzacții unice pentru verificare
primele_20_tr = tranzactii_unice[:20]
for i, tranzactie in enumerate(primele_20_tr):
    print(f"Tranzacția {i+1}: {tranzactie}")

print("Numărul de tranzacții unice:", len(tranzactii_unice))
print("Numărul de produse unice incluse în tranzacții:", len(encoder.columns_))


# Incarcare date
df_encoded = pd.read_csv('encoded_transactions.csv', dtype='int8')

# Conversie la tip bool
df_encoded = df_encoded.astype(bool)

# Definire valori suport si incredere
nivele_suport = [0.2, 0.1, 0.05, 0.01, 0.005]
nivele_incredere = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Functie pentru generarea regulilor pentru setul de iteme
def generare_reguli_itemsets(df, nivel_suport, nivele_incredere):
    print(f”Generare set de iteme frecvent pentru nivelul de suport: {nivel_suport}”)
    
    # Folosirea parametrului “memorie_scazuta” pentru seturile de date mari
    set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
    print(f”Numărul de set_iteme frecvente pentru nivelul de suport {nivel_suport}: {len(set_iteme)}")

    num_lista_regula = []
    for conf in nivele_incredere:
        print(f”Generare regule pentru nivelul de suport {nivel_suport} și încredere {conf}")
        rules = association_rules(set_iteme, metric="confidence", min_threshold=conf)
        print(f”Număr de reguli generate pentru nivelul de suport de {nivel_suport} și încredere {conf}: {len(rules)}")
        num_lista_regula.append(len(rules))
    
    return num_lista_regula

# Initializarea listelor pentru a stoca seturile de reguli

#reguli_support_20 = generare_reguli_itemsets(df_encoded, 0.2, nivele_incredere)
#reguli_support_10 = generare_reguli_itemsets(df_encoded, 0.1, nivele_incredere)
#reguli_support_5 = generare_reguli_itemsets(df_encoded, 0.05, nivele_incredere)
#reguli_support_1 = generare_reguli_itemsets(df_encoded, 0.01, nivele_incredere)
#reguli_support_0_5 = generare_reguli_itemsets(df_encoded, 0.005, nivele_incredere)

# Grafic rezultate
#plt.figure(figsize=(14, 10))

# Grafic pt fiecare nivel de incredere
#plt.plot(nivele_incredere, reguli_support_20, marker='o', label=‘Nivel suport 20%’)
#plt.plot(nivele_incredere, reguli_support_10, marker='o', label='Nivel suport 10%')
#plt.plot(nivele_incredere, reguli_support_5, marker='o', label='Nivel suport 5%')
#plt.plot(nivele_incredere, reguli_support_1, marker='o', label='Nivel suport 1%')
#plt.plot(nivele_incredere, reguli_support_0_5, marker='o', label='Nivel suport 0.5%)

#plt.xlabel(‘Nivel de încredere')
#plt.ylabel(‘Număr de reguli găsit’)
#plt.title(‘Algoritmul apriori cu diferite niveluri de suport')
#plt.legend()
#plt.grid(True)

#plt.show()

# Generare regulile graficului pentru fiecare nivel de suport
#for nivel_suport in nivele_suport:
    #num_reguli = generare_reguli_itemsets(df_encoded, nivel_suport, nivele_incredere)
    
 
    #plt.figure(figsize=(7, 5))
   # plt.plot(nivele_incredere, num_reguli, marker='o', label=f'Support level {nivel_suport*100:.1f}%')
    

    #plt.xlabel(‘Nivel de încredere')
   # plt.ylabel(‘Număr de reguli găsit')
   # plt.title(f’Algoritmul Apriori - Nivel de Suport {nivel_suport*100:.1f}%')
   # plt.legend()
   # plt.grid(True)
    
 
   # plt.show()



# # Definire niveluri suport și încredere
# nivel_suport = 0.1  # 10%
# nivele_incredere = [0.5, 0.6, 0.7]

# # Functie pentru generarea frecventei set_iteme și reguli
# def generare_reguli_itemsets(df, nivel_suport, nivel_incredere):
#     print(f”Generarea setului de iteme frecvent pentru nivelul de suport: {nivel_suport}”)
    
#     # Folosirea memorie_scazuta pentru a lucra cu seturi mari de date
#     set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
#     print(f”Numărul de seturi de iteme frecvent pentru nivelul de suport de {nivel_suport}: {len(set_iteme)}")

#     print(f”Generarea regulilor pentru nivelul de suport de {nivel_suport} și încredere de {nivel_incredere}")
#     rules = association_rules(set_iteme, metric="confidence", min_threshold=nivel_incredere)
#     print(f”Număr de reguli generate pentru nivelul de suport de  {nivel_suport} și încredere de {nivel_incredere}: {len(rules)}")
#     return rules

# # Generare regulile pentru nivelurile specificate
# reguli_incredere_50 = generare_reguli_itemsets(df_encoded, nivel_suport, 0.5)
# reguli_incredere_60 = generare_reguli_itemsets(df_encoded, nivel_suport, 0.6)
# reguli_incredere_70 = generare_reguli_itemsets(df_encoded, nivel_suport, 0.7)

# # Functie pentru afisarea regulilor

# def display_rules(rules):
#     rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
#     rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
#     rules['count'] = (rules['support'] * len(df_encoded)).astype(int)
#     return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'count']]

# print(“\nReguli pentru nivel de suport de 10% și încredere de 50%:\n”)
# print(display_rules(reguli_incredere_50))

# print("\nReguli pentru nivel de suport de 10% și încredere de 60%:\n")
# print(display_rules(reguli_incredere_60))

# print("\nReguli pentru nivel de suport de 10% și încredere de 70%:\n")
# print(display_rules(reguli_incredere_70))

# # Vizualizare reguli
# def plot_rules(rules, title):
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(x='support', y='confidence', size='lift', sizes=(20, 200), hue='lift', data=rules)
#     plt.title(title)
#     plt.xlabel('Suport')
#     plt.ylabel('Încredere')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_rules(reguli_incredere_50, 'Apriori Rules (Support: 10%, Confidence: 50%)')
# plot_rules(reguli_incredere_60, 'Apriori Rules (Support: 10%, Confidence: 60%)')
# plot_rules(reguli_incredere_70, 'Apriori Rules (Support: 10%, Confidence: 70%)')



# # Definește nivelurile de suport și încredere

# nivel_suport = 0.2  # 20%
# nivel_incredere = 0.5  # 50%

# # Functie pentru generarea frecventei set_iteme și reguli
# def generare_reguli_itemsets(df, nivel_suport, nivel_incredere):
#     print(f”Generarea setului de iteme frecvent pentru nivelul de suport: {nivel_suport}”)    

#     # Folosirea memorie_scazuta pentru a lucra cu seturi mari de date
#     set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
#     print(f”Numărul de seturi de iteme frecvent pentru nivelul de suport de {nivel_suport}: {len(set_iteme)}")

# print(f”Generarea regulilor pentru nivelul de suport de {nivel_suport} și încredere de {nivel_incredere}")
#     rules = association_rules(set_iteme, metric="confidence", min_threshold=nivel_incredere)
#     rules = rules.sort_values(by='lift', ascending=False)
#      print(f”Număr de reguli generate pentru nivelul de suport de {nivel_suport} și încredere {conf}: {len(rules)}") 
#     return rules

# Generare reguli pentru niveluri specificate de suport si incredere

# rules = generare_reguli_itemsets(df_encoded, nivel_suport, nivel_incredere)

# # Functie pentru afisarea regulilor
# def display_rules(rules):
#     return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'count']]


# rules['count'] = rules['support'] * len(df_encoded)

# print(“\n Reguli pentru nivel de suport de 20% si incredere de 50%, ordonate :\n”)
# print(display_rules(rules))

# # Vizualizarea regulilor
# def plot_rules(rules, title):
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(x='support', y='confidence', size='lift', sizes=(20, 200), hue='lift', data=rules)
#     plt.title(title)
#     plt.xlabel('Support')
#     plt.ylabel('Confidence')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_rules(rules, 'Apriori Rules (Support: 20%, Confidence: 50%)')

# # Graph (circular layout)
# def plot_circular_graph(rules, title):
#     G = nx.DiGraph()
    
#     for _, rule in rules.iterrows():
#         for antecedent in rule['antecedents']:
#             for consequent in rule['consequents']:
#                 G.add_edge(antecedent, consequent, weight=rule['lift'])

#     plt.figure(figsize=(10, 10))
#     pos = nx.circular_layout(G)
#     edges = G.edges(data=True)
#     weights = [edge[2]['weight'] for edge in edges]
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#     plt.title(title)
#     plt.show()

# plot_circular_graph(rules, 'Grafic Circular al Reguliilor de Asociere (Suport: 20%, Încredere: 50%)')

# # Grafic matrice grupat
# def plot_grouped_matrix(rules, title):
#     pivot = rules.pivot_table(index='antecedents', columns='consequents', values='lift')
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(pivot, annot=True, cmap='coolwarm')
#     plt.title(title)
#     plt.show()

# plot_grouped_matrix(rules, 'matrice grupată a reguliilor de asociere (Suport: 20%, Încredere: 50%)')




# #####predictii
# # Generăm itemset-uri frecvente
# # Generăm itemset-uri frecvente din df_encoded care ar trebui să fie deja pregătit
# frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# # Generăm reguli de asociere
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# def get_recomandari(user_ids, df, rules):
#     recomandari = {}
#     for user_id in user_ids:
#         # Extragem tranzacțiile efectuate de utilizator din DataFrame-ul original
#         iteme_utilizator = set(df[df['user_id'] == user_id]['product_name'].unique())
        
#         # Filtrează regulile pe baza itemurilor cumpărate de utilizator
#         applicable_rules = rules[rules['antecedents'].apply(lambda x: set(x).issubset(iteme_utilizator))]
#         iteme_recomandate = set()

#         # Extragem itemurile recomandate
#         for _, row in applicable_rules.iterrows():
#             iteme_recomandate.update(row['consequents'])
        
#         # Înlăturăm itemurile deja cumpărate
#         iteme_recomandate.difference_update(iteme_utilizator)

#         # Salvăm recomandările pentru fiecare utilizator
#         recomandari[user_id] = iteme_recomandate

#     return recomandari

# # Lista de utilizatori pentru care dorim recomandări
# user_ids = [2, 3, 206209, 10, 61]
# recomandari = get_recomandari(user_ids, df, rules)

# # Afișăm recomandările pentru fiecare utilizator
# for user_id in recomandari:
#     print(f"Recomandări pentru utilizatorul {user_id}: {recomandari[user_id]}")





# ######clustering

# Încărcarea datelor
path = 'ECommerce_consumer behaviour.csv'
data = pd.read_csv(path)

# Eliminarea rândurilor cu valori lipsă în 'days_since_prior_order'
data = data.dropna(subset=['days_since_prior_order'])

# Selectarea caracteristicilor relevante
caracteristici = data[['user_id', 'order_number', 'order_hour_of_day', 'days_since_prior_order', 'department_id', 'add_to_cart_order']]

# Agregarea datelor pentru fiecare utilizator
user_data = caracteristici.groupby('user_id').agg({
    'order_number': 'max',  # Numărul total de comenzi
    'order_hour_of_day': 'mean',  # Ora medie a zilei pentru comenzile plasate
    'days_since_prior_order': 'mean',  # Zilele medii de la ultima comandă
    'add_to_cart_order': 'mean',  # Ordinea medie de adăugare în coș
    'department_id': 'nunique'  # Numărul de departamente unice
}).reset_index()

# Renumirea coloanelor pentru claritate
user_data.columns = ['user_id', 'numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']

# Normalizarea datelor
scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_data[['numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']])

# Păstrarea numelor coloanelor înainte de normalizare
coloane_caracteristici = ['numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']

# Aplicarea PCA
pca = PCA()
pca.fit(scaled_features)

# Valorile proprii (eigenvalues) și variația explicativă
eigenvalues = pca.explained_variance_
explained_variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(explained_variance)
sdev = np.sqrt(eigenvalues)

# Crearea unui DataFrame pentru a vizualiza datele PCA
pca_summary_df = pd.DataFrame({'Standard Deviation': sdev,
                               'Eigenvalues': eigenvalues,
                               'Proportion of Variance': explained_variance,
                               'Cumulative Proportion': cumulative_variance},
                              index=[f'Comp{i+1}' for i in range(len(sdev))])

print(pca_summary_df)

# # Vizualizarea graficului scree
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', label='Individual variance')
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's-', label='Cumulative variance')
# plt.title('Scree Plot')
# plt.xlabel('Componente principale')
# plt.ylabel('Procentul de Varianta explicata')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Criterii de selecție ale componentelor principale
# threshold = 75  #Aceasta valoare testeaza pragul respectiv
# important_components = np.argmax(cumulative_variance >= threshold) + 1
# print(f"Numărul minim de componente pentru a explica cel puțin {threshold}% din varianță este: {important_components}")

# kaiser_rule = eigenvalues > 1
# kaiser_components = sum(kaiser_rule)
# print(f"Numărul de componente sugerat de criteriul lui Kaiser: {kaiser_components}")

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', label='Eigenvalues')
# plt.axhline(y=1, color='r', linestyle='--', label='Kaiser criterion')
# plt.title('Scree Plot- Criteriul lui Cattell')
# plt.xlabel('Principal Components')
# plt.ylabel('Eigenvalues')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Analiza paralelă
# def analiza_paralela(data, num_simulations=100):
#     num_rows, num_cols = data.shape
#     random_eigenvalues = np.zeros((num_simulations, num_cols))
#     for i in range(num_simulations):
#         random_data = np.random.normal(size=data.shape)
#         pca = PCA(n_components=num_cols)
#         pca.fit(random_data)
#         random_eigenvalues[i, :] = pca.explained_variance_
#     mean_random_eigenvalues = np.mean(random_eigenvalues, axis=0)
#     return mean_random_eigenvalues

# mean_random_eigenvalues = analiza_paralela(scaled_features)
# plt.f
# Adaugarea coloanei "cluster" la dataframe-ul user_data
user_data["cluster"] = clase


# Definirea funcțiilor de vizualizare
def plot_circular_graph(rules, title):
    G = nx.DiGraph()
    for _, rule in rules.iterrows():
        for antecedent in rule["antecedents"]:
            for consequent in rule["consequents"]:
                if rule["lift"] >= 1.5:
                    G.add_edge(antecedent, consequent, weight=rule["lift"])

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=3000, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show()

def plot_grouped_matrix(rules, title):
    pivot = rules[rules["lift"] >= 1.5].pivot_table(index="antecedents", columns="consequents", values="lift", aggfunc="mean").fillna(0)
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".2f", annot_kws={"size": 8})
    plt.title(title, fontsize=16)
    plt.xlabel("Consequents", fontsize=14)
    plt.ylabel("Antecedents", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def display_rules(rules, segment_data_len):
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules["count"] = (rules["support"] * segment_data_len).astype(int)
    return rules[["antecedents", "consequents", "support", "confidence", "lift", "count"]]

# def display_rules(rules):
#     rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
#     rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
#     rules["count"] = (rules["support"] * len(segment_data)).astype(int)
#     return rules[["antecedents", "consequents", "support", "confidence", "lift", "count"]]

# Funcția pentru generarea set_iteme și reguli
def generare_reguli_itemsets(df, nivel_suport, nivel_incredere):
    if df.empty:
        print("Data frame-ul este gol. Set de reguli imposibil de generat.")
        return pd.DataFrame()
    
    print(f"Generare set_iteme frecvente pentru nivelul de suport de{nivel_suport}")
    set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
    if set_iteme.empty:
        print("Niciun set de iteme frecvente găsit")
        return pd.DataFrame()
    
    print(f"Numărul de set_iteme frecvent pentru nivelul de suport de {nivel_suport}: {len(set_iteme)}")
    print(f"Generare reguli pentru nivelul de suport de {nivel_suport} și încredere {nivel_incredere}")
    rules = association_rules(set_iteme, metric="confidence", min_threshold=nivel_incredere)
    if rules.empty:
        print("Nicio regulă generată.")
        return pd.DataFrame()
    
    rules = rules[rules["lift"] > 1.5]  # Filtrare pentru lift > 1.5
    rules = rules.sort_values(by="lift", ascending=False)
    print(f"Număr de reguli generate pentru nivelul de suport {nivel_suport} și încredere {nivel_incredere}: {len(rules)}")
    return rules

########################################





# # Definirea funcțiilor de vizualizare
# def plot_circular_graph(rules, title):
#     G = nx.DiGraph()
#     for _, rule in rules.iterrows():
#         for antecedent in rule['antecedents']:
#             for consequent in rule['consequents']:
#                 if rule['lift'] >= 1.5:
#                     G.add_edge(antecedent, consequent, weight=rule['lift'])

#     plt.figure(figsize=(14, 14))
#     pos = nx.spring_layout(G, k=2)
#     weights = [G[u][v]['weight'] for u, v in G.edges()]
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     plt.show()

# def plot_grouped_matrix(rules, title):
#     pivot = rules[rules['lift'] >= 1.5].pivot_table(index='antecedents', columns='consequents', values='lift', aggfunc='mean').fillna(0)
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, fmt='.2f', annot_kws={'size': 8})
#     plt.title(title, fontsize=16)
#     plt.xlabel('Consequents', fontsize=14)
#     plt.ylabel('Antecedents', fontsize=14)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.show()

# def display_rules(rules):
#     rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
#     rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
#     rules['count'] = (rules['support'] * len(segment_data)).astype(int)
#     return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'count']]

# # Funcția pentru generarea set_iteme și reguli
# def generare_reguli_itemsets(df, nivel_suport, nivel_incredere):
#     if df.empty:
#         print("Data frame-ul este gol. Set de reguli imposibil de generat.")
#         return pd.DataFrame()
    
#     print(f"Generare set_iteme frecvente pentru nivelul de suport de{nivel_suport}")
#     set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
#     if set_iteme.empty:
#         print("Niciun set de iteme frecvente găsit")
#         return pd.DataFrame()
    
#     print(f"Numărul de set_iteme frecvent pentru nivelul de suport de {nivel_suport}: {len(set_iteme)}")
#     print(f"Generare reguli pentru nivelul de suport de {nivel_suport} și încredere {nivel_incredere}")
#     rules = association_rules(set_iteme, metric="confidence", min_threshold=nivel_incredere)
#     if rules.empty:
#         print("Nicio regulă generată.")
#         return pd.DataFrame()
    
#     rules = rules[rules['lift'] > 1.5]  # Filtrare pentru lift > 1.5
#     rules = rules.sort_values(by='lift', ascending=False)
#     print(f"Număr de reguli generate pentru nivelul de suport {nivel_suport} și încredere {nivel_incredere}: {len(rules)}")
#     return rules



###cluster
# Încărcarea și pregătirea datelor
path = 'ECommerce_consumer behaviour.csv'
data = pd.read_csv(path)
data = data.dropna(subset=['days_since_prior_order'])

# Selectarea și agregarea caracteristicilor relevante
caracteristici = data[['user_id', 'order_number', 'order_hour_of_day', 'days_since_prior_order', 'department_id', 'add_to_cart_order', 'product_name']]
user_data = caracteristici.groupby('user_id').agg({
    'order_number': 'max',
    'order_hour_of_day': 'mean',
    'days_since_prior_order': 'mean',
    'add_to_cart_order': 'mean',
    'department_id': 'nunique'
}).reset_index()
user_data.columns = ['user_id', 'numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']

# Normalizarea datelor
scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_data[['numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']])

# Aplicarea PCA
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(scaled_features)
user_data['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(pca_scores)



# Generarea regulilor de asociere pentru fiecare cluster
nivel_suport = 0.1
nivel_incredere = 0.5
all_rules = {}
for cluster in user_data['cluster'].unique():
    segment_data = data[data['user_id'].isin(user_data[user_data['cluster'] == cluster]['user_id'])]
    basket = segment_data.groupby(['user_id', 'product_name'])['add_to_cart_order'].sum().unstack().reset_index().fillna(0).set_index('user_id')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    rules = generare_reguli_itemsets(basket, nivel_suport, nivel_incredere)
    if not rules.empty:
        all_rules[cluster] = rules
        
        print(f"Reguli de asociere pentru clusterul {cluster}:\n")
        display(display_rules(rules))
        
        plot_grouped_matrix(rules, f'Matrice grupată a reguliilor de asociere pentru clusterul {cluster}')
        plot_circular_graph(rules, f'Grafic circular al reguliilor de asociere pentru clusterul {cluster}')
    else:
        print(f"Nicio regula generata pentru cluster {cluster}.")



# Funcții de vizualizare
def plot_circular_graph(rules, title):
    G = nx.DiGraph()
    for _, rule in rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                if rule['lift'] >= 1.2:
                    G.add_edge(antecedent, consequent, weight=rule['lift'])

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=2, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], edge_color=weights, edge_cmap=plt.cm.Blues, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='black', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=10, font_color='black')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.colorbar(nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Blues, alpha=0.7))
    plt.show()

def plot_grouped_matrix(rules, title):
    pivot = rules[rules['lift'] >= 1.2].pivot_table(index='antecedents', columns='consequents', values='lift', aggfunc='mean').fillna(0)
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, fmt='.2f', annot_kws={'size': 8})
    plt.title(title, fontsize=16)
    plt.xlabel('Consequents', fontsize=14)
    plt.ylabel('Antecedents', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def display_rules(rules):
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Funcția pentru generarea set_iteme și reguli
def generare_reguli_itemsets(df, nivel_suport, nivel_incredere):
    if df.empty:
        print("Data frame gol. Imposibil de generat reguli.")
        return pd.DataFrame()
    
    print(f"Generare set_iteme frecvente pentru nivelul de suport de {nivel_suport}")
    set_iteme = apriori(df, min_support=nivel_suport, use_colnames=True, memorie_scazuta=True)
    if set_iteme.empty:
        print("No frequent set_iteme found.")
        return pd.DataFrame()
    
    print(f"Number of frequent set_iteme for support level {nivel_suport}: {len(set_iteme)}")
    print(f"Generating rules for support level {nivel_suport} and confidence {nivel_incredere}")
    rules = association_rules(set_iteme, metric="confidence", min_threshold=nivel_incredere)
    if rules.empty:
        print("No rules generated.")
        return pd.DataFrame()
    
    rules = rules[rules['lift'] > 1.2]  # Filtrare pentru lift > 1.2
    rules = rules.sort_values(by='lift', ascending=False)
    print(f"Number of rules generated for support level {nivel_suport} and confidence {nivel_incredere}: {len(rules)}")
    return rules

# Încărcarea și pregătirea datelor
path = 'ECommerce_consumer behaviour.csv'
data = pd.read_csv(path)
data = data.dropna(subset=['days_since_prior_order'])

# Selectarea și agregarea caracteristicilor relevante
caracteristici = data[['user_id', 'order_number', 'order_hour_of_day', 'days_since_prior_order', 'department_id', 'add_to_cart_order', 'product_name']]
user_data = caracteristici.groupby('user_id').agg({
    'order_number': 'max',
    'order_hour_of_day': 'mean',
    'days_since_prior_order': 'mean',
    'add_to_cart_order': 'mean',
    'department_id': 'nunique'
}).reset_index()
user_data.columns = ['user_id', 'numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']

# Normalizarea datelor
scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_data[['numar_total_comenzi', 'ora_medie_comenzi', 'media_zilelor_ultima_c', 'nr_mediu_produse', 'nr_departamente']])

# Aplicarea PCA
pca = PCA(n_components=3)
pca_scores = pca.fit_transform(scaled_features)
user_data['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(pca_scores)

# Generarea regulilor de asociere pentru fiecare cluster
nivel_suport = 0.1  # Reduced support level to 0.1
nivel_incredere = 0.5
all_rules = {}
for cluster in user_data['cluster'].unique():
    segment_data = data[data['user_id'].isin(user_data[user_data['cluster'] == cluster]['user_id'])]
    basket = segment_data.groupby(['user_id', 'product_name'])['add_to_cart_order'].sum().unstack().reset_index().fillna(0).set_index('user_id')
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    rules = generare_reguli_itemsets(basket, nivel_suport, nivel_incredere)
    if not rules.empty:
        all_rules[cluster] = rules
        
        print(f"Reguli de asociere pentru clusterul {cluster}:\n")
        display(display_rules(rules))
        
        plot_grouped_matrix(rules, f'Matrice grupată a reguliilor de asociere pentru clusterul {cluster}')
        plot_circular_graph(rules, f'Grafic circular al reguliilor de asociere pentru clusterul {cluster}')
    else:
        print(f"Nicio regulă generată pentru clusterul {cluster}.")