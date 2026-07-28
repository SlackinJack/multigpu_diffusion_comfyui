# starting on 07.27.2026, all requirements
# will be installed to the project dir's .venv


echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r .venv_tf4 .venv_tf5 multigpu_diffusion
find -iname "__pycache__" | xargs rm -r


echo ""
echo "########## Clone multigpu_diffusion ##########"
echo ""
git clone https://github.com/slackinjack/multigpu_diffusion --depth=1


echo ""
echo "########## Setup multigpu_diffusion ##########"
echo ""
cd multigpu_diffusion
bash setup.sh
cd ..


echo ""
echo "########## Setup .venvs (transformers 4.x.x) ##########"
echo ""
cp multigpu_diffusion/tf_4_requirements.txt req_multi.txt
cp multigpu_diffusion/AsyncDiff/requirements.txt req_async.txt
python3 -m venv .venv_tf4
source .venv_tf4/bin/activate && pip install --no-cache-dir -r req_async.txt
source .venv_tf4/bin/activate && pip install --no-cache-dir -r req_multi.txt
rm req_async.txt req_multi.txt


echo ""
echo "########## Setup .venvs (transformers 5.x.x) ##########"
echo ""
cp multigpu_diffusion/tf_5_requirements.txt req_multi.txt
cp multigpu_diffusion/AsyncDiff/requirements.txt req_async.txt
python3 -m venv .venv_tf5
source .venv_tf5/bin/activate && pip install --no-cache-dir -r req_async.txt
source .venv_tf5/bin/activate && pip install --no-cache-dir -r req_multi.txt
rm req_async.txt req_multi.txt


echo ""
echo "########## Setup complete, enjoy! ##########"
echo ""
