# Hello there 👋

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jordandeklerk.jordandeklerk)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataScientist(nn.Module):
    def __init__(self):
        super(DataScientist, self).__init__()
        self.name = "Jordan Deklerk"
        self.role = "Senior Data Scientist"
        self.company = "DICK's Sporting Goods"
        self.experience = ["Retail", "Healthcare"]
        self.programming = ["Python", "R", "SQL", "SAS", "STATA"]
        self.tools = ["Azure ML", "AWS Sagemaker", "Databricks", "Spark", "Docker", "Kubeflow", "GCP"]

        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.exp_embed = nn.Embedding(len(self.experience), 64)
        self.prog_embed = nn.Embedding(len(self.programming), 64)
        self.tool_embed = nn.Embedding(len(self.tools), 64)

    def self_attention(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention_logits = torch.einsum('bik,bjk->bij', q, k) / (64 ** 0.5)
        attention_weights = F.softmax(attention_logits, dim=-1)
        attended = torch.einsum('bij,bjk->bik', attention_weights, v)
        
        return attended

    def forward(self, focus):
        focus_dict = {"experience": 0, "programming": 1, "tools": 2}
        focus_idx = focus_dict.get(focus, 0)
        
        exp_emb = self.exp_embed(torch.arange(len(self.experience)))
        prog_emb = self.prog_embed(torch.arange(len(self.programming)))
        tool_emb = self.tool_embed(torch.arange(len(self.tools)))
        all_emb = torch.cat([exp_emb, prog_emb, tool_emb], dim=0).unsqueeze(0) 

        attended = self.self_attention(all_emb)
        focused = attended[0, focus_idx]  
        
        return focused

    def say_hi(self):
        print("Thanks for dropping by, hope you find some of my work interesting.")

me = DataScientist()
me.say_hi()
```

## 📝 Website and Socials

- Personal website: [jordandeklerk.com](https://jordandeklerk.com)
- LinkedIn: [linkedin.com/in/jordandeklerk](https://www.linkedin.com/in/jordandeklerk)

## 📔 Latest Machine Learning Posts

<!-- BLOG-POST-LIST:START -->
- [An Introduction to Reinforcement Learning](https://ml-tutorials.netlify.app/blog/rl-intro/)
- [Masked Token Learning for Inpatient Diagnosis and Procedure Prediction](https://ml-tutorials.netlify.app/blog/ehr-bert/)
- [Closing the Amortization Gap in Bayesian Deep Generative Models](https://ml-tutorials.netlify.app/blog/amortized-bayes/)
<!-- BLOG-POST-LIST:END -->

## 🔧 Technologies & Tools

**Cloud Services:**

![Azure](https://img.shields.io/badge/Azure-0089D6?style=flat&logo=microsoft-azure&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white) ![GCP](https://img.shields.io/badge/GCP-4285F4?style=flat&logo=google-cloud&logoColor=white)

**Programming Languages:**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-336791?style=flat&logo=postgresql&logoColor=white) ![SAS](https://img.shields.io/badge/SAS-0066B8?style=flat&logo=sas&logoColor=white) ![STATA](https://img.shields.io/badge/STATA-1D91C2?style=flat&logo=stata&logoColor=white)

**Tools and Services:**

![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![Databricks](https://img.shields.io/badge/Databricks-FC4C02?style=flat&logo=databricks&logoColor=white) ![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apache-spark&logoColor=white)

## 🗂️ Highlight Projects

<a href="https://github.com/jordandeklerk/EHR-BERT">
  <img align="center" src="https://github-readme-stats.vercel.app/api/pin/?username=jordandeklerk&repo=EHR-BERT&show_icons=true&line_height=27&title_color=6aa6f8&text_color=8a919a&icon_color=6aa6f8&bg_color=22272e" alt="EHR-BERT" />
</a>

<a href="https://github.com/jordandeklerk/Amortized-Bayes">
  <img align="center" src="https://github-readme-stats.vercel.app/api/pin/?username=jordandeklerk&repo=Amortized-Bayes&show_icons=true&line_height=27&title_color=6aa6f8&text_color=8a919a&icon_color=6aa6f8&bg_color=22272e" alt="Amortized Bayes" />
</a>

<a href="https://github.com/jordandeklerk/SwinViT">
  <img align="center" src="https://github-readme-stats.vercel.app/api/pin/?username=jordandeklerk&repo=SwinViT&show_icons=true&line_height=27&title_color=6aa6f8&text_color=8a919a&icon_color=6aa6f8&bg_color=22272e" alt="SwinViT" />
</a>
