import pandas as pd

# Lista de colunas a serem excluídas
colunas_a_excluir = [
    'ldrmodules.not_in_mem',
    'pslist.avg_threads',
    'svcscan.process_services',
    'ldrmodules.not_in_init',
    'handles.nsemaphore',
    'pslist.nppid',
    'handles.nfile',
    'ldrmodules.not_in_mem_avg',
    'ldrmodules.not_in_init_avg',
    'svcscan.nactive',
    'psxview.not_in_deskthrd',
    'callbacks.ncallbacks',
    'malfind.commitCharge',
    'ldrmodules.not_in_load_avg',
    'handles.ndesktop',
    'handles.ndirectory',
    'pslist.nproc',
    'malfind.uniqueInjections',
    'malfind.ninjections',
    'malfind.protection',
    'psxview.not_in_csrss_handles_false_avg',
    'psxview.not_in_csrss_handles',
    'psxview.not_in_deskthrd_false_avg',
    'psxview.not_in_session_false_avg',
    'psxview.not_in_session',
    'svcscan.fs_drivers',
    'callbacks.nanonymous',
    'pslist.nprocs64bit',
    'handles.nport',
    'svcscan.interactive_process_services',
    'psxview.not_in_pslist',
    'psxview.not_in_eprocess_pool',
    'psxview.not_in_pspcid_list',
    'psxview.not_in_ethread_pool',
    'psxview.not_in_pslist_false_avg',
    'modules.nmodules',
    'psxview.not_in_pspcid_list_false_avg',
    'psxview.not_in_ethread_pool_false_avg',
    'psxview.not_in_eprocess_pool_false_avg',
    'callbacks.ngeneric'
]

# Lendo o arquivo CSV
df = pd.read_csv('/content/Obfuscated-MalMem2022.csv')

# Excluindo as colunas indesejadas
df_cleaned = df.drop(columns=colunas_a_excluir, errors='ignore')

# Salvando o novo DataFrame em um novo arquivo CSV
df_cleaned.to_csv('/content/Obfuscated-MalMem2022.csv', index=False)

print("Colunas excluídas com sucesso!")
