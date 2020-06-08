
## **Intro to Data Science with Ames Housing Dataset**

In this blog post, we will take a bird eye view of various steps involved in a data science project, we'll get from Exploraratory Data Analysis to Feature Engineering  and Handling of Outliers to Model Evaluation, so sit tight this is gonna be an exciting read !!

**so, What is Data Science  ?**
In a nutshell Data Science has mostly to do with analysis of data through modelling and conducting experiments for doing inference or prediction.
Still data science is different than statistics, in some sense, due to its ability to work on qualitative data (e.g. images and text) as well.
After the advent of Information age, while digital data is omnipresent now, data science seems to have the capability to solve problems which were not possible earlier.  
> “Data science has become a fourth approach to scientific discovery, in addition to experimentation, modeling, and computation,” said **Provost Martha Pollack**

[50 years of Data Science](https://courses.csail.mit.edu/18.337/2015/docs/50YearsDataScience.pdf)

**Data**
The Data contains scores of countries based on parameters like, Gross Development Product or GDP, Freedom, Happiness, Corruption and Life Expectancy.
Using these parameters a rating has been created for every country, namely Happiness Score this rating defines how much the conditions differ for  people living in different countries.
Data has been collected for 4 years, each corresponding to year from 2015 to 2018, new parameters have been added to recent year's data

 **Problem**
 * Does Corruption affects Happinesss?

* What is happiness score in case of poor countries (Low GDP) ?

* What makes a Country more happy?
 * Economy (GDP) or Freedom

* What relates most to Happiness?

```python
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
```




```python
#Reading Files
df_2015 = pd.read_csv('./894_813759_bundle_archive/2015.csv')
df_2016 = pd.read_csv('./894_813759_bundle_archive/2016.csv')

df_2017 = pd.read_csv('./894_813759_bundle_archive/2017.csv')
df_2018 = pd.read_csv('./894_813759_bundle_archive/2018.csv')
```


```python
#Lets take a look at columns
df_2015.columns,df_2015.shape
```




    (Index(['Country', 'Happiness Rank', 'Happiness Score',
            'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
            'Freedom', 'Trust (Government Corruption)', 'Generosity'],
           dtype='object'),
     (158, 9))




```python
df_2018.columns,df_2018.shape
```




    (Index(['Overall rank', 'Country or region', 'Score', 'GDP per capita',
            'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity',
            'Perceptions of corruption'],
           dtype='object'),
     (156, 9))



## Data Cleaning

As we can see there is some inconsistency in columns between individual daaframes, we need to ensure that same features are used while concatenating them.


```python
df_2015.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Standard Error</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.587</td>
      <td>0.03411</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.41978</td>
      <td>0.29678</td>
      <td>2.51738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.561</td>
      <td>0.04884</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.14145</td>
      <td>0.43630</td>
      <td>2.70201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.527</td>
      <td>0.03328</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.48357</td>
      <td>0.34139</td>
      <td>2.49204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.522</td>
      <td>0.03880</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.36503</td>
      <td>0.34699</td>
      <td>2.46531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>North America</td>
      <td>5</td>
      <td>7.427</td>
      <td>0.03553</td>
      <td>1.32629</td>
      <td>1.32261</td>
      <td>0.90563</td>
      <td>0.63297</td>
      <td>0.32957</td>
      <td>0.45811</td>
      <td>2.45176</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2016.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Lower Confidence Interval</th>
      <th>Upper Confidence Interval</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.526</td>
      <td>7.460</td>
      <td>7.592</td>
      <td>1.44178</td>
      <td>1.16374</td>
      <td>0.79504</td>
      <td>0.57941</td>
      <td>0.44453</td>
      <td>0.36171</td>
      <td>2.73939</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.509</td>
      <td>7.428</td>
      <td>7.590</td>
      <td>1.52733</td>
      <td>1.14524</td>
      <td>0.86303</td>
      <td>0.58557</td>
      <td>0.41203</td>
      <td>0.28083</td>
      <td>2.69463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.501</td>
      <td>7.333</td>
      <td>7.669</td>
      <td>1.42666</td>
      <td>1.18326</td>
      <td>0.86733</td>
      <td>0.56624</td>
      <td>0.14975</td>
      <td>0.47678</td>
      <td>2.83137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.498</td>
      <td>7.421</td>
      <td>7.575</td>
      <td>1.57744</td>
      <td>1.12690</td>
      <td>0.79579</td>
      <td>0.59609</td>
      <td>0.35776</td>
      <td>0.37895</td>
      <td>2.66465</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>5</td>
      <td>7.413</td>
      <td>7.351</td>
      <td>7.475</td>
      <td>1.40598</td>
      <td>1.13464</td>
      <td>0.81091</td>
      <td>0.57104</td>
      <td>0.41004</td>
      <td>0.25492</td>
      <td>2.82596</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.632</td>
      <td>1.305</td>
      <td>1.592</td>
      <td>0.874</td>
      <td>0.681</td>
      <td>0.202</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Norway</td>
      <td>7.594</td>
      <td>1.456</td>
      <td>1.582</td>
      <td>0.861</td>
      <td>0.686</td>
      <td>0.286</td>
      <td>0.340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Denmark</td>
      <td>7.555</td>
      <td>1.351</td>
      <td>1.590</td>
      <td>0.868</td>
      <td>0.683</td>
      <td>0.284</td>
      <td>0.408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.495</td>
      <td>1.343</td>
      <td>1.644</td>
      <td>0.914</td>
      <td>0.677</td>
      <td>0.353</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Switzerland</td>
      <td>7.487</td>
      <td>1.420</td>
      <td>1.549</td>
      <td>0.927</td>
      <td>0.660</td>
      <td>0.256</td>
      <td>0.357</td>
    </tr>
  </tbody>
</table>
</div>



We will take columns in df_2015 as baseline, as we can see df_2016 is having few extra columns, so we will remove them.


```python
df_2016.drop(['Lower Confidence Interval','Upper Confidence Interval','Region','Dystopia Residual'],axis=1,inplace=True)
```


```python
df_2015.drop(['Standard Error','Region','Dystopia Residual'],axis=1,inplace=True)
```

Lets clean rest of dataframes.


```python
df_2015.columns
```




    Index(['Country', 'Happiness Rank', 'Happiness Score',
           'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
           'Freedom', 'Trust (Government Corruption)', 'Generosity'],
          dtype='object')




```python
df_2017.head()  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
  </tbody>
</table>
</div>




```python
updated_columns = [i.replace('.',' ') for i in df_2017.columns.tolist()]

df_2017.columns = updated_columns
```


```python
df_2017.drop([ 'Whisker high','Whisker low','Dystopia Residual'],axis=1,inplace = True)
```


```python
df_2017.rename(columns={'Economy  GDP per Capita ':'Economy (GDP per Capita)',
                   'Health  Life Expectancy ':'Health (Life Expectancy)',
                  'Trust  Government Corruption ':'Trust (Government Corruption)'},inplace=True)
```


```python
df_2018
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness Rank</th>
      <th>Country</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust (Government Corruption)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.632</td>
      <td>1.305</td>
      <td>1.592</td>
      <td>0.874</td>
      <td>0.681</td>
      <td>0.202</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Norway</td>
      <td>7.594</td>
      <td>1.456</td>
      <td>1.582</td>
      <td>0.861</td>
      <td>0.686</td>
      <td>0.286</td>
      <td>0.340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Denmark</td>
      <td>7.555</td>
      <td>1.351</td>
      <td>1.590</td>
      <td>0.868</td>
      <td>0.683</td>
      <td>0.284</td>
      <td>0.408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.495</td>
      <td>1.343</td>
      <td>1.644</td>
      <td>0.914</td>
      <td>0.677</td>
      <td>0.353</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Switzerland</td>
      <td>7.487</td>
      <td>1.420</td>
      <td>1.549</td>
      <td>0.927</td>
      <td>0.660</td>
      <td>0.256</td>
      <td>0.357</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>Yemen</td>
      <td>3.355</td>
      <td>0.442</td>
      <td>1.073</td>
      <td>0.343</td>
      <td>0.244</td>
      <td>0.083</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>Tanzania</td>
      <td>3.303</td>
      <td>0.455</td>
      <td>0.991</td>
      <td>0.381</td>
      <td>0.481</td>
      <td>0.270</td>
      <td>0.097</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>South Sudan</td>
      <td>3.254</td>
      <td>0.337</td>
      <td>0.608</td>
      <td>0.177</td>
      <td>0.112</td>
      <td>0.224</td>
      <td>0.106</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155</td>
      <td>Central African Republic</td>
      <td>3.083</td>
      <td>0.024</td>
      <td>0.000</td>
      <td>0.010</td>
      <td>0.305</td>
      <td>0.218</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>Burundi</td>
      <td>2.905</td>
      <td>0.091</td>
      <td>0.627</td>
      <td>0.145</td>
      <td>0.065</td>
      <td>0.149</td>
      <td>0.076</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 9 columns</p>
</div>




```python
df_2018.rename(columns={'Country or region':'Country',
                        'Score':'Happiness Score',
                   'Overall rank':'Happiness Rank',
                        'GDP per capita':'Economy (GDP per Capita)',
                        'Freedom to make life choices':'Freedom',
                   'Healthy life expectancy':'Health (Life Expectancy)',
                        'Social support':'Family',
                  'Perceptions of corruption':'Trust (Government Corruption)'},inplace=True)
```


```python
df = pd.concat([df_2015,df_2016,df_2017,df_2018],axis=0).reset_index(drop=True)
```

## Exploratory Data Analysis


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>626.000000</td>
      <td>626.000000</td>
      <td>626.000000</td>
      <td>626.000000</td>
      <td>626.000000</td>
      <td>626.000000</td>
      <td>625.000000</td>
      <td>626.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.747604</td>
      <td>5.372021</td>
      <td>0.918764</td>
      <td>1.045891</td>
      <td>0.584299</td>
      <td>0.415706</td>
      <td>0.129138</td>
      <td>0.226981</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.219609</td>
      <td>1.131774</td>
      <td>0.409808</td>
      <td>0.328946</td>
      <td>0.241948</td>
      <td>0.154943</td>
      <td>0.108202</td>
      <td>0.126854</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.693000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.000000</td>
      <td>4.497750</td>
      <td>0.606755</td>
      <td>0.847945</td>
      <td>0.404142</td>
      <td>0.310500</td>
      <td>0.056565</td>
      <td>0.137263</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>79.000000</td>
      <td>5.307000</td>
      <td>0.983705</td>
      <td>1.081274</td>
      <td>0.632553</td>
      <td>0.434635</td>
      <td>0.094000</td>
      <td>0.208581</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118.000000</td>
      <td>6.187250</td>
      <td>1.239502</td>
      <td>1.283387</td>
      <td>0.772957</td>
      <td>0.538998</td>
      <td>0.161570</td>
      <td>0.290915</td>
    </tr>
    <tr>
      <th>max</th>
      <td>158.000000</td>
      <td>7.632000</td>
      <td>2.096000</td>
      <td>1.644000</td>
      <td>1.030000</td>
      <td>0.724000</td>
      <td>0.551910</td>
      <td>0.838075</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df);
```
![plot](/_posts/output_23_0.png)


# Questions we will answer:

* Does <u>Corruption  affects Happinesss?


* What <u>makes a Country more happy?
 * Economy (GDP) or Freedom


* What is <u> happiness score in case of poor countries (Low GDP) ?


* What <u>relates most to Happiness?


## Does Corruption affects Happinesss?


```python
x = 'Trust (Government Corruption)'
y = 'Happiness Score'
sns.jointplot(data=df,x=x,y=y, kind='reg');
```


![plot](/output_26_0.png)


### No, Corruption and Happiness are not strongly correlated

## So, What makes a Country more happy ?


```python
x = 'Freedom'
y = 'Happiness Score'
sns.lmplot(data=df,x=x,y=y);
```


![png](/output_29_0.png)



```python
x = 'Economy (GDP per Capita)'
y = 'Happiness Score'
sns.lmplot(data=df,x=x,y=y);
```


![png](/output_30_0.png)


### As visible in above graphs, it is clear that Economy of a country has stronger correlation with Happiness than Freedom.

**Notes:**
* Freedom still has some weak correlation with Happiness


## What is happiness score in case of poor countries (Low GDP) ?

We'll consider the countries having gdp score less than mean (~0.9) as poor.


```python
df_poor = df[df['Economy (GDP per Capita)'] < df['Economy (GDP per Capita)'].mean()]

#filter rich countries
df_rich = df[~(df['Economy (GDP per Capita)'] < df['Economy (GDP per Capita)'].mean())]
```


```python
df_poor['Happiness Score'].mean()
```




    4.551802119851534




```python
df_rich['Happiness Score'].mean()
```




    6.04876093360634



### The mean happiness score for poor countries is significantly less than rich countries.
* Hence, it can be said that countries with less gdp often tend to have less happiness score.

## What relates most to Happiness?


```python
X = df[['Economy (GDP per Capita)','Family', 'Health (Life Expectancy)','Freedom', 'Trust (Government Corruption)','Generosity']]

y = df[['Happiness Score']]
```


```python
X.isnull().sum()
```




    Economy (GDP per Capita)         0
    Family                           0
    Health (Life Expectancy)         0
    Freedom                          0
    Trust (Government Corruption)    1
    Generosity                       0
    dtype: int64



We have missing value in one of our columns, we will impute it with mean.


```python
X = X.fillna(X.mean())
```


```python
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
```

We will split our data into train and test split, while using 33% of data as test set.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

We will create a model and fit train data.


```python
reg = linear_model.LinearRegression(normalize=True)
```


```python
reg.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)




```python
y_pred = reg.predict(X_test)
```

**r2 score** is a metric used to measure the accuracy of our predictions w.r.t the true value.

More close r2 score is to 1, more our predictions are closer to actual values.

[More on r2 metric here](https://en.wikipedia.org/wiki/Coefficient_of_determination)


```python
r2_score(y_test, y_pred)
```




    0.7289187955037795




```python
cols = X.columns.tolist()

weight = reg.coef_.tolist()[0]
```


```python
for i in range(len(X.columns)):
    print(cols[i],' : ' ,weight[i]) 
```

    Economy (GDP per Capita)  :  1.0030019299874529
    Family  :  0.7154479296823543
    Health (Life Expectancy)  :  1.222139514991989
    Freedom  :  1.4426167236646017
    Trust (Government Corruption)  :  0.9890544070126698
    Generosity  :  0.4447904262072287


### Freedom has the highest coefficient in our linear refression model, hence it relates most with Happiness score.
