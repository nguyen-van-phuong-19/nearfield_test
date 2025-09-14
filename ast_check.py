import ast
code=open('main_simulation_gui.py','r',encoding='utf-8',errors='replace').read()
try:
    ast.parse(code)
    print('AST OK')
except Exception as e:
    print('AST ERROR:',e)
