echo -e "\033[1;31m copy utils/thulac/models \033[0m"
echo -e "\033[1;31m download Git Graph \033[0m"

git config --global user.name "Ichinoe"
git config --global user.email "ShiinaHiiragi@outlook.com"

pip install simpletransformers

python utils/saver.py
python PromptWeaver.py --dataset=msra --pclass=bart --suffix=train
python PromptWeaver.py --dataset=msra --pclass=entail --suffix=train
