from math import log #importa a rotina log da biblioteca math

#Função para Calcular o Valor Final
def valor_final():
    ci=float(input("Entre com o Capital inicial: "))
    i=float(input("Entre com a taxa: "))
    t=int(input("Entre período: "))
    vf = ci * (1 + i)**t
    print("------------------------------------------")
    print(f"O valor final é R$ {vf:,.2f}")
    print("==========================================")

#Função para Calcular capital inicial  
def capital_inicial():
    vf=float(input("Entre com o Valor Final: "))
    i=float(input("Entre com a taxa: "))
    t=float(input("Entre período: "))
    ci = vf / (1 + i)**t
    print("------------------------------------------")
    print(f"O Capital Inicial é R$ {ci:,.2f}")
    print("==========================================")

#Função para Calcular período  
def periodo():
    vf=float(input("Entre com o Valor Final: "))
    ci=float(input("Entre com o Capital inicial: "))
    i=float(input("Entre com a taxa: "))
    t = int(log(vf / ci) / log(1 + i))
    print("------------------------------------------")
    print(f"O período é de {t:,.2f}")
    print("==========================================")

#Função para Calcular Taxa 
def taxa():
    vf=float(input("Entre com o Valor Final: "))
    ci=float(input("Entre com o Capital inicial: "))
    t=int(input("Entre período: "))
    i = ((vf / ci)**(1/t)) - 1
    print("------------------------------------------")
    print(f"A taxa é de {i:,.2f}")
    print("==========================================")

def opcao_invalida():
    print("Opção inválida. Por favor, escolha uma opção conforme Menu.")

# Faz um Menu de opção para organizar os pedidos
def menu():
    while True:
        print("Bem-vindo à calculadora de Juros Composto")
        print("O que você gostaria de calcular?")
        print("1 - Valor Final (vf)")
        print("2 - Capital Inicial (ci)")
        print("3 - Período (t)")
        print("4 - Taxa (i)")
        print ("0 - SAIR")
        print("--------------------------------------------")
        try:
            opcao=int(input("Digite o número de sua opção: ")) # variável opção do Menu
            # Rotina para chamar a opção retirando possíveis erros

            if opcao>0: 
                switcher = {   # Dicionário de opção do Menu, uma forma simplificada para chamar as funções
                    1: valor_final,
                    2: capital_inicial,
                    3: periodo,
                    4: taxa,
                }
                func = switcher.get(opcao, opcao_invalida)
                func()  # Chama a função def baseada na opcao

                while True:
                    continua = input("Deseja mais algum cálculo (S) para sim (N) para Não: ")
                    if continua.upper()=="N":
                       opcao=0
                       break
                    elif continua.upper()!="S":
                        print("Opcao errada!!")
                    else:
                        break
                
            if opcao== 0:
                print("Se precisar fazer mais cálculos, volte a me executar.")
                break
            else:
                print("Formato Incorreto")
        except:
            print("Erro")
    

menu()
