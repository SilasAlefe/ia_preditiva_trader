import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"Inicialização falhou, código de erro: {mt5.last_error()}")
else:
    print("Conexão bem sucedida")
mt5.shutdown()