echo "copy utils/thulac/models"
echo "download Git Graph"

git config --global user.name "Ichinoe"
git config --global user.email "ShiinaHiiragi@outlook.com"

python PromptWeaver.py --dataset=msra --pclass=bart --suffix=train
python PromptWeaver.py --dataset=msra --pclass=entail --suffix=train
python utils/saver.py

pip install simpletransformers
