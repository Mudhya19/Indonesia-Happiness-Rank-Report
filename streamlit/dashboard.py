# Imoorting Library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import warnings
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot


warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
plt.style.use('Solarize_Light2')

@st.cache_resource
def load_data():
    df_2021 = pd.read_csv('..\data\world-happiness-report-2021.csv')
    df_past = pd.read_csv('..\data\world-happiness-report.csv')
    return df_2021, df_past

df_2021,df_past = load_data()

st.title('Indonesia Happiness Rank Dashboard Report')

with st.sidebar:
    st.image("children.png")

if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Raw Data")
    st.write(df_2021)

# Display summary statistics
# Display summary statistics
# Display summary statistics
# Display summary statistics
if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(df_past.describe())

col1, col2 = st.columns(2)

SEA = df_2021[df_2021['Regional indicator'] == "Southeast Asia"]['Country name'].to_list()

print("Negara-negara yang ada dalam ASEAN terdiri dibawah ini : \n")

pd.DataFrame(SEA)

colors_blue = ["#132C33", "#264D58", '#17869E', '#51C4D3', '#B4DBE9']
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_mix = ["#17869E", '#264D58', '#179E66', '#D35151', '#E9DAB4', '#E9B4B4', '#D3B651', '#6351D3']
colors_div = ["#132C33", '#17869E', '#DADADA', '#D35151', '#331313']

sns.palplot(colors_blue)
sns.palplot(colors_dark)
sns.palplot(colors_red)
sns.palplot(colors_mix)
sns.palplot(colors_div)

def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb =mpl.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict

th = [0, 0.2, 0.5, 0.8, 1]
cdict = NonLinCdict(th, colors_blue)
cdiv = NonLinCdict(th, colors_div)

cm = LinearSegmentedColormap('blue', cdict)
cm_div = LinearSegmentedColormap('div', cdiv)

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.titlecolor'] = colors_dark[0]
mpl.rcParams['axes.labelcolor'] = colors_dark[0]

# tick
mpl.rcParams['xtick.color'] = colors_dark[0]
mpl.rcParams['ytick.color'] = colors_dark[0]
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


# legend 
mpl.rcParams['legend.edgecolor'] = colors_dark[0]

with col1:
    st.subheader("Regional Indicator Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_2021, x="Regional indicator", ax=ax)
    plt.xticks(rotation=60)
    st.pyplot(fig)
with col2: 
    st.subheader("Happiest and Unhappiest Countries in 2021")
    df2021_happiest_unhappiest = df_2021[(df_2021["Ladder score"] > 7.4) | (df_2021["Ladder score"] < 3.5)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x="Ladder score", y="Country name", data=df2021_happiest_unhappiest, palette="magma", ax=ax)
    ax.set_title("Happiest and Unhappiest Countries in 2021")
    st.pyplot(fig)

# 2021 wolrd Happiest and Unhappiest
low_c = '#dd4124'
high_c = '#009473'
plt.rcParams["font.family"] = "monospace"

fig = plt.figure(figsize=(6,3),dpi=150)
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.2, hspace=0.4)
ax0 = fig.add_subplot(gs[0, 0])

background_color = "#fafafa"
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(1.167,0.85,"2021 World Happiness Index",color='#323232',fontsize=28, fontweight='bold', fontfamily='sanserif',ha='center')
ax0.text(1.13,-0.35,"stand-out facts",color='lightgray',fontsize=28, fontweight='bold', fontfamily='monospace',ha='center')

ax0.text(0,0.4,"Finland",color=high_c,fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0,0.1,"Happiest",color='gray',fontsize=15, fontfamily='monospace',ha='center')

ax0.text(0.77,0.4,"9 of top 10",color=high_c,fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(0.75,0.1,"in Europe",color='gray',fontsize=15, fontfamily='monospace',ha='center')

ax0.text(1.5,0.4,"7 of bottom 10",color=low_c,fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(1.5,0.1,"in Africa",color='gray',fontsize=15, fontfamily='monospace',ha='center')

ax0.text(2.25,0.4,"Afghanistan",color=low_c,fontsize=25, fontweight='bold', fontfamily='monospace',ha='center')
ax0.text(2.25,0.1,"Unhappiest",color='gray',fontsize=15, fontfamily='monospace',ha='center')

ax0.set_yticklabels('')
ax0.set_xticklabels('')
ax0.tick_params(axis='both',length=0)

for s in ['top','right','left','bottom']:
    ax0.spines[s].set_visible(False)
    
import matplotlib.lines as lines
l1 = lines.Line2D([0.15, 1.95], [0.67, 0.67], transform=fig.transFigure, figure=fig,color = 'gray', linestyle='-',linewidth = 1.1, alpha = .5)
fig.lines.extend([l1])
l2 = lines.Line2D([0.15, 1.95], [0.07, 0.07], transform=fig.transFigure, figure=fig,color = 'gray', linestyle='-',linewidth = 1.1, alpha = .5)
fig.lines.extend([l2])
    
st.pyplot(fig)

#Life Ladder Comparison By Countries
fig = px.choropleth(df_past.sort_values("year"), 
                    locations = "Country name", 
                    color = "Life Ladder",
                    locationmode = "country names",
                    animation_frame = "year")
fig.update_layout(title = "Life Ladder Comparison by Countries")
st.plotly_chart(fig, use_container_width=True, width=800)

# Plotting happiness score distribution
df = df_2021[df_2021['Country name'].isin(SEA)]
df_2021_top = df_2021.iloc[0:1]
df_2021_bot = df_2021.iloc[-1]
mean_score = df_2021['Ladder score'].mean()
sea_idx = list(df.index + 1)

fig, ax = plt.subplots(figsize=(14, 8))
bars0 = ax.bar(df_2021_top['Country name'], df_2021_top['Ladder score'], color=colors_blue[0], alpha=0.6, edgecolor=colors_dark[0])
bars1 = ax.bar(df['Country name'], df['Ladder score'], color=colors_dark[3], alpha=0.4, edgecolor=colors_dark[0])
bars2 = ax.bar(df_2021_bot['Country name'], df_2021_bot['Ladder score'], color=colors_red[0], alpha=0.6, edgecolor=colors_dark[0])
line  = ax.axhline(mean_score, linestyle='--', color=colors_dark[2])

ax.legend(["Average", "Happiest", "SEA Countries", "Unhappiest"], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, borderpad=1, frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
ax.set_xlabel("Countries", fontsize=14, labelpad=10, fontweight='bold', color=colors_dark[0])
ax.set_ylabel("Ladder Score", fontsize=14, labelpad=10, fontweight='bold', color=colors_dark[0])
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

avgl  = ax.text(
    s="Global\nAverage: {:.2f}".format(mean_score),
    x=xmax*1.02,
    y=mean_score,
    backgroundcolor=colors_dark[2],
    fontsize=12,
    fontweight='bold',
    color='white'
)
# Indonesia settings
bars1[5].set_alpha(1)
bars1[5].set_color(colors_red[3])
bars1[5].set_edgecolor(colors_dark[0])

for i, bar in enumerate(bars1) : 
    x=bar.get_x()
    y=bar.get_height()
    if i != 5 : 
        ax.text(
            s=f"{sea_idx[i]}th",
            va='center', ha='center', 
            x=x+0.38, y=y/2,
            color=colors_dark[3],
            fontsize=14,
        )
    else : 
        ax.text(
        s=f"{sea_idx[i]}th",
        va='center', ha='center', 
        x=x+0.38, y=y/2,
        color='white',
        fontsize=14,
        fontweight='bold'
    )
        
for i, bar in enumerate(bars0) : 
    x=bar.get_x(),
    y=bar.get_height(),

    ax.text(
        s=f"1st",
        va='center', ha='center', 
        x=x[0]+0.38, y=y[0]/2,
        color="white",
        fontsize=14,
        fontweight='bold',
        alpha=1,
    )
    
for i, bar in enumerate(bars2) : 
    x=bar.get_x(),
    y=bar.get_height(),

    ax.text(
        s="149th",
        va='center', ha='center', 
        x=x[0]+0.38, y=y[0]/2,
        color="white",
        fontsize=14,
        fontweight='bold',
        alpha=1,
    )
        
plt.text(s="How Happy is Indonesia Among its Neighbors?", ha='left', x=xmin, y=ymax*1.12, fontsize=24, fontweight='bold', color=colors_dark[0])
plt.title("Among SEA countries, Indonesia ranks 6th in the happiness index.\nIn a world context, Indonesia ranks 82th, still falls below average.", loc='left', fontsize=13, color=colors_dark[2])  
plt.tight_layout()
plt.show()

# Tampilkan chart menggunakan Streamlit
st.pyplot(fig)
# Kesimpulan
st.subheader("Kesimpulan dari Grafik")
st.write("""
- **Posisi Indonesia**:
    - Indonesia berada di peringkat 82 dalam indeks kebahagiaan global, yang berada di bawah rata-rata global.
    - Di antara negara-negara Asia Tenggara, Indonesia berada di posisi tengah (peringkat 6 dari 10 negara yang disebutkan).
- **Rata-rata Kebahagiaan**:
    - Rata-rata global kebahagiaan adalah 5.53, dan skor kebahagiaan Indonesia berada di bawah rata-rata ini.
- **Pentingnya Konteks Regional dan Global**:
    - Meski Indonesia berada di peringkat menengah dalam konteks Asia Tenggara, dalam konteks global, Indonesia masih berada di bawah rata-rata kebahagiaan.
""")

# menambahkan Kontribusi Faktor-faktor terhadap kebahagian di indonesias
st.header("menambahkan Kontribusi Faktor-faktor terhadap kebahagian di indonesias")
# Filter data untuk Indonesia
indonesia_data = df_2021[df_2021['Country name'] == 'Indonesia']

# Pilih kolom-kolom yang relevan
factors = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 
           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

# Analisis kontribusi masing-masing faktor terhadap skor kebahagiaan di Indonesia
indonesia_factors = indonesia_data[factors].transpose()
indonesia_factors.columns = ['Contribution']
indonesia_factors = indonesia_factors.sort_values(by='Contribution', ascending=False)

# Visualisasikan kontribusi faktor-faktor di Indonesia
fig , ax = plt.subplots(figsize=(10,6))
sns.barplot(x=indonesia_factors['Contribution'], y=indonesia_factors.index, palette='viridis')
ax.set_title('Kontribusi Faktor-faktor terhadap Kebahagiaan di Indonesia')
ax.set_xlabel('Kontribusi')
ax.set_ylabel('Faktor')
st.pyplot(fig)

st.write(f"Beberapa faktor-faktor terhadap Kebahagiaan di Indonesia : ", indonesia_factors)

#menambahkan distribution faktor-faktor terhadap kebahagiaan di Indonesia# Distribusi Kontribusi Faktor-faktor terhadap Kebahagiaan di Indonesia
st.subheader("Distribusi Kontribusi Faktor-faktor terhadap Kebahagiaan di Indonesia")

def freedman_diaconis_bins(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    bins = round((data.max() - data.min()) / bin_width)
    return bins

# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 9))
# fig.suptitle('Distribusi Kontribusi Faktor-faktor terhadap Kebahagiaan di Indonesia', y=1.02, fontsize=16)

# Flatten the axes for easy iteration
axes = axes.flatten()

# Loop through numeric columns and create histograms
for i, col in enumerate(factors):
    bins = freedman_diaconis_bins(df_2021[col])
    sns.histplot(df_2021[col], kde=True, ax=axes[i], bins=bins)
    axes[i].set_title(f'Distribusi {col}')

# Adjust layout to prevent overlap
plt.tight_layout()
st.pyplot(fig)

#kesimpulan 
st.subheader("Kesimpulan dari Grafik")
st.write("""
**Kesimpulan**:
- Healthy life expectancy dan Logged GDP per capita adalah dua faktor utama yang paling berkontribusi terhadap kebahagiaan di Indonesia. Hal ini menandakan pentingnya kesehatan dan kesejahteraan ekonomi dalam meningkatkan kebahagiaan masyarakat.
- Faktor-faktor lain seperti kebebasan membuat pilihan hidup, persepsi terhadap korupsi, dukungan sosial, dan kedermawanan juga penting tetapi memiliki kontribusi yang lebih kecil.
- Untuk meningkatkan kebahagiaan secara keseluruhan, kebijakan yang berfokus pada peningkatan kesehatan masyarakat dan kesejahteraan ekonomi perlu diprioritaskan. Selain itu, upaya untuk meningkatkan kebebasan pribadi, mengurangi korupsi, memperkuat dukungan sosial, dan mendorong kedermawanan juga akan memberikan dampak positif.""")