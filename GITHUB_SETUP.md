# Push to GitHub

## 1. Create a new repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Name it `ercot-market-intelligence` (or any name)
3. Leave it empty (no README, no .gitignore)
4. Create the repo

## 2. Push from your machine

```bash
cd "/Users/SaiDinesh/Documents/market intelligence"

# Add your GitHub repo as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ercot-market-intelligence.git

# Push
git push -u origin main
```

If you use SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/ercot-market-intelligence.git
git push -u origin main
```

## 3. Run on Google Colab

1. Open https://colab.research.google.com
2. **File → Open notebook → GitHub** tab
3. Paste: `https://github.com/YOUR_USERNAME/ercot-market-intelligence`
4. Open `ercot_colab.ipynb`
5. In the first cell, the repo is already cloned (Colab does this when opening from GitHub)
6. In the second cell, set `EIA_API_KEY = "your-key"` (get one at https://www.eia.gov/opendata/)
7. Run all cells

Or use Colab Secrets: **Keychain icon → Add new secret** → name `EIA_API_KEY`, then in the notebook:
```python
from google.colab import userdata
EIA_API_KEY = userdata.get('EIA_API_KEY')
```
