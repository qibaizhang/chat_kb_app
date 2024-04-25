
from server import app, server

from dash.dependencies import Input, Output, State
import feffery_markdown_components as fmc
import feffery_utils_components as fuc
import feffery_antd_components as fac
from datetime import datetime
from dash import html, dcc
import dash

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.llms import SparkLLM
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.text import TextLoader

# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())    # read local .env file


 # # 如果使用原生openai接口，可把这个注释取消，设置代理，根据你的实际情况调整端口
 # # 若是可以直连的第三方接口的话，直接在界面填一下api_base_url和api_key即可
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def gen_spark_params(model):
    '''
    构造星火模型请求参数
    '''

    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
        }
    }
    return model_params_dict[model]

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
def generate_response(question, llm, *args):
    # llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key, base_url=os.environ.get("OPENAI_API_BASE"))
    # llm = ChatZhipuAI(
    #     temperature=0.01,
    #     api_key=zhipuai_api_key,
    #     model_name="glm-4",
    # )
    output = llm.invoke(question)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output

def get_vectordb(embedding):
    # 定义 Embeddings
    # embedding = embedding
    # 向量数据库持久化路径
    persist_directory = 'vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

# #带有历史记录的问答链
# def get_chat_qa_chain(question, llm, embedding, *args):
#     vectordb = get_vectordb(embedding)
#     # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key, base_url=os.environ.get("OPENAI_API_BASE"))  
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
#         return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
#     )
#     retriever=vectordb.as_retriever()
#     qa = ConversationalRetrievalChain.from_llm(
#         llm,
#         retriever=retriever,
#         memory=memory
#     )
#     result = qa({"question": question})
#     return result['answer']

#不带历史记录的问答链
def get_qa_chain(question, llm, template, embedding):
    vectordb = get_vectordb(embedding)
    # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key, base_url=os.environ.get("OPENAI_API_BASE"))

    # template = template
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


app.layout = fac.AntdWatermark(
    [
        # 注入问题返回状态消息提示
        html.Div(
            id='response-status-message'
        ),

        # 注入历史对话记录存储
        dcc.Store(
            id='multi-round-store',
            data={
                'status': '关闭',
                'history': []
            }
        ),
        
        dcc.Store(
            id='key-api-store',
            data={
                'openai': {
                    'api_key': '',
                    'api_base': 'https://api.openai.com/v1',
                    'model_name': 'gpt-3.5-turbo'
                },
                '智谱': {
                    'api_key': '',
                    'model_name': 'glm-4'
                },
                '星火': {
                    'appid': '',
                    'api_key': '',
                    'api_secret': '',
                    'model_name': 'v3.5'
                },
                '文心': {
                    'api_key': '',
                    'secret_key': '',
                    'model_name': 'Yi-34B-Chat'
                }
            }
        ),

        # # 注入问答记录markdown下载
        # dcc.Download(
        #     id='history-qa-records-download'
        # ),

        fac.AntdRow(
            [
                fac.AntdCol(
                    html.Div(
                        fuc.FefferyDiv(
                            [
                                html.Div(
                                    fac.AntdParagraph(
                                        [
                                            fac.AntdText(
                                                '选择大模型',
                                                strong=True,
                                                italic=True,
                                                style={
                                                    'fontSize': 22
                                                }
                                            ),
                                            fac.AntdText(
                                                '（请填入自己的key和参数）',
                                                type='secondary',
                                                style={
                                                    'fontSize': 10
                                                }
                                            )
                                        ]
                                    ),
                                    style={
                                        # 'padding': '15px 0'
                                    }                                    
                                ),
                                html.Div(
                                    fac.AntdSelect(
                                        id='model-select',
                                        placeholder='请选择',
                                        options=[
                                            {
                                                'label': '智谱',
                                                'value': '智谱'
                                            },
                                            {
                                                'label': 'openai',
                                                'value': 'openai'
                                            },
                                            {
                                                'label': '星火',
                                                'value': '星火'
                                            },
                                            {
                                                'label': '文心',
                                                'value': '文心'
                                            },
                                        ],
                                        defaultValue='智谱',
                                        style={
                                            'width': '40%',
                                        }
                                    ),
                                    style={
                                        'padding': '0 0 15px 0'
                                    }
                                ),
                                html.Div(id='model-params-value'),
                                fac.AntdRadioGroup(
                                    options=[
                                        {
                                            'label': '直接模型对话',
                                            'value': '直接模型对话'
                                        },
                                        {
                                            'label': '带知识库对话',
                                            'value': '带知识库对话'
                                        },
                                    ],
                                    id='select-chat-type',
                                    defaultValue='直接模型对话'
                                ),
                                html.Div(
                                    [
                                        fac.AntdParagraph(
                                            [
                                                fac.AntdText(
                                                    '知识库对话提示词',
                                                    strong=True,
                                                    italic=True,
                                                    style={
                                                        'fontSize': 22
                                                    }
                                                ),
                                                fac.AntdText(
                                                    '（直接对话可在中间输入框直接写提示词）',
                                                    type='secondary',
                                                    style={
                                                        'fontSize': 10
                                                    }
                                                )
                                            ]
                                        ),
                                        fac.AntdInput(
                                            id='prompt-template-input',
                                            mode='text-area',
                                            autoSize=False,
                                            allowClear=True,
                                            placeholder='请输入提示词模板：',
                                            defaultValue=template,
                                            # size='large',
                                            style={
                                                'fontSize': 16,
                                                'height': '180px'
                                            }
                                        )
                                    ],
                                    style={
                                        'padding': '15px 0'
                                    } 
                                )                                 
                            ],
                            shadow='always-shadow',
                            className='chat-wrapper1'
                        ),
                        className='root-wrapper1'
                        # style={
                        #     'backgroundColor': 'rgba(64, 173, 255, 1)',
                        #     'color': 'white',
                        #     'height': '100px',
                        #     'display': 'flex',
                        #     'justifyContent': 'center',
                        #     'alignItems': 'center'
                        # }
                    ),
                    span=7
                ),
                fac.AntdCol(
                    html.Div(
                        fuc.FefferyDiv(
                            [
                                fac.AntdRow(
                                    [
                                        fac.AntdCol(
                                            fac.AntdParagraph(
                                                [
                                                    fac.AntdText(
                                                        'AI大模型知识库对话v0.0.1',
                                                        strong=True,
                                                        italic=True,
                                                        style={
                                                            'fontSize': 22
                                                        }
                                                    ),
                                                    fac.AntdText(
                                                        '（开发完善中）',
                                                        type='secondary',
                                                        style={
                                                            'fontSize': 10
                                                        }
                                                    )
                                                ]
                                            )
                                        ),

                                        fac.AntdCol(
                                            fac.AntdSpace(
                                                [
                                                    fac.AntdFormItem(
                                                        fac.AntdSwitch(
                                                            id='enable-multi-round',
                                                            checked=False,
                                                            checkedChildren='开启',
                                                            unCheckedChildren='关闭'
                                                        ),
                                                        label='多轮对话',
                                                        style={
                                                            'marginBottom': 0
                                                        }
                                                    ),
                                                    fac.AntdTooltip(
                                                        fac.AntdButton(
                                                            id='export-history-qa-records',
                                                            icon=fac.AntdIcon(
                                                                icon='antd-save'
                                                            ),
                                                            type='primary',
                                                            shape='circle'
                                                        ),
                                                        title='保存当前全部对话记录并生成下载链接'
                                                    ),
                                                    fac.AntdTooltip(
                                                        fac.AntdButton(
                                                            id='download-history-qa-records',
                                                            icon=fac.AntdIcon(
                                                                icon='antd-download'
                                                            ),
                                                            type='primary',
                                                            shape='circle'
                                                        ),
                                                        title='点击下载当前全部对话记录(先点旁边的保存再下载)'
                                                    ),
                                                    fac.AntdTooltip(
                                                        fac.AntdButton(
                                                            id='clear-exists-records',
                                                            icon=fac.AntdIcon(
                                                                icon='antd-clear'
                                                            ),
                                                            type='primary',
                                                            shape='circle',
                                                            danger=True
                                                        ),
                                                        title='一键清空当前对话'
                                                    )
                                                ]
                                            )
                                        )
                                    ],
                                    justify='space-between'
                                ),

                                # 聊天记录容器
                                html.Div(
                                    [
                                        fac.AntdSpace(
                                            [
                                                fac.AntdAvatar(
                                                    mode='icon',
                                                    icon='antd-robot',
                                                    style={
                                                        'background': '#1890ff'
                                                    }
                                                ),
                                                fuc.FefferyDiv(
                                                    fac.AntdText(
                                                        '你好，欢迎使用基于ai大模型服务的在线知识库对话机器人😋',
                                                        style={
                                                            'fontSize': 16
                                                        }
                                                    ),
                                                    className='chat-record-container'
                                                )
                                            ],
                                            align='start',
                                            style={
                                                'padding': '10px 15px',
                                                'width': '100%'
                                            }
                                        )
                                    ],
                                    id='chat-records'
                                ),

                                # 聊天输入区
                                fac.AntdSpace(
                                    [
                                        fac.AntdInput(
                                            id='new-question-input',
                                            mode='text-area',
                                            autoSize=False,
                                            allowClear=True,
                                            placeholder='请输入问题：',
                                            size='large',
                                            style={
                                                'fontSize': 16
                                            }
                                        ),
                                        fac.AntdButton(
                                            '提交',
                                            id='send-new-question',
                                            type='primary',
                                            block=True,
                                            autoSpin=True,
                                            loadingChildren='思考中',
                                            size='large'
                                        )
                                    ],
                                    direction='vertical',
                                    size=2,
                                    style={
                                        'width': '100%'
                                    }
                                )
                            ],
                            shadow='always-shadow',
                            className='chat-wrapper'
                        ),
                        className='root-wrapper'
                    ),
                    span=10
                ),
                fac.AntdCol(
                    html.Div(
                        fuc.FefferyDiv(
                            html.Div(
                                [
                                    html.Div(id='succes-result1'),
                                    html.H2(
                                        "上传知识库文件",
                                        style={
                                            'font-weight': 'bolder'
                                            , 'color': 'rgb(64 126 255)'
                                            , 'textAlign': 'center'
                                        }
                                    ),

                                    fac.AntdDraggerUpload(
                                        id='upload',
                                        apiUrl='/upload/',
                                        uploadId='my-files',
                                        fileTypes=['pdf', 'md'],
                                        multiple=True,
                                        # fileMaxSize=1,
                                        failedTooltipInfo='啊哦，上传过程出了问题...',
                                        showUploadList=False,
                                        text='上传pdf,md文件',
                                        hint='点击或拖拽文件至此处进行上传，可多选',
                                        style={
                                            'maxHeight': '500px'
                                        }
                                    ),
                                    fac.AntdDivider(
                                        '选择向量模型',
                                        isDashed=True
                                    ),
                                    fac.AntdSpace(
                                        [
                                            fac.AntdSelect(
                                                id='embedding-select',
                                                placeholder='请选择embedding模型',
                                                options=[
                                                    {
                                                        'label': '智谱:embedding-2',
                                                        'value': '智谱:embedding-2'
                                                    },
                                                    {
                                                        'label': 'openai:text-embedding-3-small',
                                                        'value': 'openai:text-embedding-3-small'
                                                    },
                                                    {
                                                        'label': 'openai:text-embedding-3-large',
                                                        'value': 'openai:text-embedding-3-large'
                                                    },
                                                    {
                                                        'label': 'openai:text-embedding-ada-002',
                                                        'value': 'openai:text-embedding-ada-002'
                                                    },
                                                ],
                                                defaultValue='智谱:embedding-2',
                                                style={
                                                    # 'width': '40%',
                                                }
                                            ),                                         
                                            fac.AntdButton(
                                                [
                                                    fac.AntdIcon(
                                                        icon='md-fingerprint'
                                                    ),
                                                    '向量入库'
                                                ],
                                                id="start-embedding-button",
                                                type='primary',
                                                loadingChildren='入库中',
                                                autoSpin=True                                                
                                            ),
                                        ],
                                        size=40,
                                        addSplitLine=True,
                                        style={
                                            'padding': '10px',
                                            'height': '100px',
                                            # 'marginLeft': '130px',
                                            # 'gap': '100px',
                                            'display': 'flex',
                                            'justifyContent': 'center',

                                        }
                                    )
                                    ,
                                    fac.AntdDivider(
                                        '操作流程',
                                        isDashed=True
                                    ),
                                    fac.AntdSteps(
                                        steps=[
                                            {
                                                'title': '上传知识库文件',
                                                'description': '选择符合格式要求的文件上传',
                                            },
                                            {
                                                'title': '选择向量模型',
                                                'description': '要在左上角选择模型厂商并输入key',
                                            },
                                            {
                                                'title': '点击按钮向量入库',
                                                'description': '需要时间，等待向量入库完成',
                                            },
                                        ],
                                        allowClick=True,
                                        size='small',
                                        current=0,
                                        labelPlacement='vertical'
                                    )

                                ],
                                style={
                                    # 'minHeight': '300px',
                                    # 'width': '1000px',
                                    # 'boxShadow': 'rgb(79 89 108 / 51%) 1px 4px 15px',
                                    # 'borderRadius': '2px',
                                    # 'padding': '25px'

                                }
                            ),
                            shadow='always-shadow',
                            className='chat-wrapper1'
                        ),
                        className='root-wrapper1'
                        # style={
                        #     'backgroundColor': 'rgba(64, 173, 255, 1)',
                        #     'color': 'white',
                        #     'height': '100px',
                        #     'display': 'flex',
                        #     'justifyContent': 'center',
                        #     'alignItems': 'center'
                        # }
                    ),
                    span=7
                ),
            ],
            # gutter=10
        ),
        
    ],
    content=''
)


@app.callback(
    [Output('start-embedding-button', 'loading'), 
     Output('succes-result1', 'children')], 
    Input('start-embedding-button', 'nClicks'),
    [State('upload', 'lastUploadTaskRecord'),
     State('embedding-select', 'value'),
     State('key-api-store', 'data')]
)
def embedding_button(nClicks, lastUploadTaskRecord, value, api_store):
    if nClicks: 
        if lastUploadTaskRecord:
            try:
                file_paths = []
                for l in lastUploadTaskRecord:    
                    inpath = os.path.join(
                        'caches',
                        l['taskId'],
                        l['fileName']
                    )
                    file_paths.append(inpath)
                # 遍历文件路径并把实例化的loader存放在loaders里
                # print(file_paths)
                loaders = []            
                for file_path in file_paths:
                    file_type = file_path.split('.')[-1]
                    if file_type == 'pdf':
                        loaders.append(PyMuPDFLoader(file_path))
                    elif file_type == 'md':
                        loaders.append(UnstructuredMarkdownLoader(file_path))
                    # elif file_type == 'csv':
                    #     print(file_path)
                    #     try:
                    #         loaders.append(CSVLoader(file_path))
                    #     except Exception as e:
                    #         print(e)
                    # elif file_type == 'txt':
                    #     print(file_path)
                    #     try:
                    #         loaders.append(TextLoader(file_path))
                    #     except Exception as e:
                    #         print(e)
                # 下载文件并存储到text
                texts = []
                for loader in loaders: texts.extend(loader.load())
                # 切分文档
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50)
                split_docs = text_splitter.split_documents(texts)   
                emb_co = value.split(":")[0]
                emb_model = value.split(":")[1]
                if emb_co == '智谱':
                    os.environ["ZHIPUAI_API_KEY"] = api_store['智谱']['api_key']
                    embedding = ZhipuAIEmbeddings()
                elif emb_co == 'openai':
                    embedding = OpenAIEmbeddings(
                        api_key=api_store['openai']['api_key'],
                        base_url=api_store['openai']['api_base'],
                        model=emb_model
                    )
                # elif emb_co == '文心':
                #     embedding = QianfanEmbeddingsEndpoint()
                # 定义持久化路径
                persist_directory = 'vector_db/chroma'
                vectordb = Chroma.from_documents(
                    documents=split_docs, 
                    embedding=embedding,
                    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
                )
                vectordb.persist()
            except Exception as e:
                # print(e)
                return [
                    False, 
                    fac.AntdMessage(
                    content=f'入库失败！{e}',
                    duration=2,
                    type='warning')
                ]
            return [
                False, 
                fac.AntdMessage(
                content='入库成功！',
                duration=2,
                type='success')
            ]
    return dash.no_update


@app.callback(
    Output('model-params-value', 'children'),
    Input('model-select', 'value'),
    prevent_initial_call=True
)
def input_model_parama(model_value):
    if model_value == '智谱':
        return fac.AntdForm(
            [
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_key',
                        placeholder='请输入api_key',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_key'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='model_name',
                        placeholder='例如glm-4',
                        # defaultValue='glm-4',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='model_name'
                ),
            ],
            id='model-para-form',
            enableBatchControl=True,
            values={
                'api_key': '',
                'model_name': 'glm-4'
            }
            # style={
            #     'width': '300px',
            #     'margin': '0 auto'  # 以快捷实现居中布局效果
            # }
        )
    elif model_value == 'openai':
        return fac.AntdForm(
            [
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_key',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_key'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_base',
                        placeholder='例如https://api.openai.com/v1',
                        # defaultValue='https://api.openai.com/v1',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_base'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='model_name',
                        placeholder='例如gpt-3.5-turbo',
                        # defaultValue='gpt-3.5-turbo',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='model_name'
                ),
            ],
            id='model-para-form',
            enableBatchControl=True,
            values={
                'api_key': '',
                'api_base': 'https://api.openai.com/v1',
                'model_name': 'gpt-3.5-turbo'
            }
            # style={
            #     'width': '300px',
            #     'margin': '0 auto'  # 以快捷实现居中布局效果
            # }
        )
    elif model_value == '星火':
        return fac.AntdForm(
            [
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='appid',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='appid'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_key',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_key'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_secret',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_secret'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='model_name',
                        placeholder='例如v1.5,v2.0,v3.0,v3.5',
                        # defaultValue='v3.5',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='model_name'
                ),
            ],
            id='model-para-form',
            enableBatchControl=True,
            values={
                'appid': '',
                'api_key': '',
                'api_secret': '',
                'model_name': 'v3.5'
            }
            # style={
            #     'width': '300px',
            #     'margin': '0 auto'  # 以快捷实现居中布局效果
            # }
        )
    elif model_value == '文心':
        return fac.AntdForm(
            [
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='api_key',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='api_key'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='secret_key',
                        placeholder='请输入',
                        mode='password',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='secret_key'
                ),
                fac.AntdFormItem(
                    fac.AntdInput(
                        name='model_name',
                        placeholder='例如ERNIE-Bot-turbo',
                        # defaultValue='ERNIE-Bot-turbo',
                        style={
                            'width': '100%'
                        }
                    ),
                    label='model_name'
                ),
            ],
            id='model-para-form',
            enableBatchControl=True,
            values={
                'api_key': '',
                'secret_key': '',
                'model_name': 'Yi-34B-Chat'
            }
            # style={
            #     'width': '300px',
            #     'margin': '0 auto'  # 以快捷实现居中布局效果
            # }
        )
    
    return dash.no_update


@app.callback(
    Output('key-api-store', 'data'),
    Input('model-para-form', 'values'),
    [State('model-select', 'value'),
     State('key-api-store', 'data')],
    prevent_initial_call=True
)
def store_api_key(api_values, model_value, api_store_values):
    if model_value == '智谱' and api_values.get('api_key'):
        api_store_values['智谱']['api_key'] = api_values.get('api_key')
        api_store_values['智谱']['model_name'] = api_values.get('model_name')
        return api_store_values
    elif model_value == 'openai' and api_values.get('api_key') and api_values.get('api_base'):
        api_store_values['openai']['api_key'] = api_values.get('api_key')
        api_store_values['openai']['api_base'] = api_values.get('api_base')
        api_store_values['openai']['model_name'] = api_values.get('model_name')
        return api_store_values
    elif model_value == '星火' and api_values.get('appid') and api_values.get('api_key') and api_values.get('api_secret'):
        api_store_values['星火']['appid'] = api_values.get('appid')
        api_store_values['星火']['api_key'] = api_values.get('api_key')
        api_store_values['星火']['api_secret'] = api_values.get('api_secret')
        api_store_values['星火']['model_name'] = api_values.get('model_name')
        return api_store_values
    elif model_value == '文心' and api_values.get('api_key') and api_values.get('secret_key'):
        api_store_values['文心']['api_key'] = api_values.get('api_key')
        api_store_values['文心']['secret_key'] = api_values.get('secret_key')
        api_store_values['文心']['model_name'] = api_values.get('model_name')
        return api_store_values
    # print(api_store_values)
    return dash.no_update


@app.callback(
    [Output('chat-records', 'children'),
     Output('new-question-input', 'value'),
     Output('send-new-question', 'loading'),
     Output('response-status-message', 'children'),
     Output('multi-round-store', 'data')],
    [Input('send-new-question', 'nClicks'),
     Input('clear-exists-records', 'nClicks'),
     Input('enable-multi-round', 'checked')],
    [State('new-question-input', 'value'),
     State('chat-records', 'children'),
     State('multi-round-store', 'data'),
     State('model-select', 'value'),
     State('model-para-form', 'values'),
     State('select-chat-type', 'value'),
     State('embedding-select', 'value'),
     State('key-api-store', 'data'),
     State('prompt-template-input', 'value')],
    prevent_initial_call=True
)
def send_new_question(new_question_trigger,
                      clear_records_trigger,
                      enable_multi_round,
                      question,
                      origin_children,
                      multi_round_store,
                      model_value,
                      model_para,
                      chat_type,
                      embedding_value,
                      api_store,
                      template):
    '''
    控制以渲染或清空对话框内容为目的的操作，包括处理新问题的发送、已有记录的清空、多轮对话模式的切换等
    '''

    # print(model_para)
    emb_co = embedding_value.split(":")[0]
    emb_model = embedding_value.split(":")[1]
    
    if emb_co == '智谱':
        os.environ["ZHIPUAI_API_KEY"] = api_store['智谱']['api_key']
        embedding = ZhipuAIEmbeddings()
    elif emb_co == 'openai':
        embedding = OpenAIEmbeddings(
            api_key=api_store['openai']['api_key'],
            base_url=api_store['openai']['api_base'],
            model=emb_model
        ) 
           
    if model_value == '智谱':
        zhipuai_api_key = api_store['智谱']['api_key']
        llm = ChatZhipuAI(
            temperature=0.01,
            api_key=zhipuai_api_key,
            model_name=model_para['model_name'],
        )
    elif model_value == 'openai':
        llm = ChatOpenAI(temperature=0, 
                         base_url=api_store['openai']['api_base'],
                         model=model_para['model_name'],
                         api_key=api_store['openai']['api_key']
        )
    elif model_value == '星火':
        spark_api_url = gen_spark_params(model_para['model_name'])["spark_url"]
        domain = gen_spark_params(model_para['model_name'])["domain"]
        
        llm = SparkLLM(temperature=0.01, 
                        spark_app_id=api_store['星火']['appid'],
                        spark_api_key=api_store['星火']['api_key'],
                        spark_api_secret=api_store['星火']['api_secret'],
                        spark_api_url=spark_api_url,
                        spark_llm_domain=domain,
        )
    elif model_value == '文心':
        llm = QianfanLLMEndpoint(temperature=0.01, 
                         qianfan_ak=api_store['文心']['api_key'],
                         model=model_para['model_name'],
                         qianfan_sk=api_store['文心']['secret_key']
        )
    
    if chat_type == '直接模型对话':
        model_response = generate_response
    else:
        model_response = get_qa_chain
        
    # 若当前回调由提交新问题触发
    if dash.ctx.triggered_id == 'send-new-question' and new_question_trigger:

        # 检查问题输入是否有效
        if not question:
            return [
                dash.no_update,
                dash.no_update,
                False,
                fac.AntdMessage(
                    content='请完善问题内容后再进行提交！',
                    type='warning'
                ),
                dash.no_update
            ]

        # 尝试将当前的问题发送至大模型问答服务接口
        try:
            
            messages=(
                # 若当前模式为多轮对话模式，则附带上历史对话记录以维持对话上下文
                [
                    *(
                        multi_round_store.get('history') or []
                    ),
                    {"role": "user", "content": question}
                ]
                if enable_multi_round
                else [
                    {"role": "user", "content": question}
                ]
            )

            if len(messages) == 1:
                send_message = messages[0].get('content')
            else:
                last_message = messages[-1].get('content')
                send_message = f'这是这次要问的问题：{last_message}，这是之前历史对话：{str(messages[:-1])}，请给出回答。'

            response = model_response(send_message, llm, template, embedding)


        except Exception as e:
            return [
                dash.no_update,
                dash.no_update,
                False,
                fac.AntdMessage(
                    content='回复生成失败，错误原因：'+str(e),
                    type='error'
                ),
                dash.no_update
            ]

        # 将上一次历史问答记录中id为latest-response-begin的元素过滤掉
        origin_children = [
            child
            for child in origin_children
            if child['props'].get('id') != 'latest-response-begin'
        ]

        # 更新各输出目标属性
        return [
            [
                *origin_children,
                # 渲染当前问题
                fac.AntdSpace(
                    [
                        fac.AntdAvatar(
                            mode='text',
                            text='我',
                            style={
                                'background': '#1890ff'
                            }
                        ),
                        fuc.FefferyDiv(
                            fac.AntdText(
                                question,
                                copyable=True,
                                style={
                                    'fontSize': 16
                                }
                            ),
                            className='chat-record-container',
                            style={
                                'maxWidth': 680
                            }
                        )
                    ],
                    align='start',
                    style={
                        'padding': '10px 15px',
                        'width': '100%',
                        'flexDirection': 'row-reverse'
                    }
                ),
                # 在当前问题回复之前注入辅助滚动动作的目标点
                html.Div(
                    id='latest-response-begin'
                ),
                # 渲染当前问题的回复
                fac.AntdSpace(
                    [
                        fac.AntdAvatar(
                            mode='icon',
                            icon='antd-robot',
                            style={
                                'background': '#1890ff'
                            }
                        ),
                        fuc.FefferyDiv(
                            fmc.FefferyMarkdown(
                                markdownStr=response,
                                codeTheme='okaidia',
                                codeFallBackLanguage='python'  # 遇到语言不明的代码块，统统视作python渲染
                            ),
                            className='chat-record-container',
                            style={
                                'maxWidth': 680
                            }
                        )
                    ],
                    align='start',
                    style={
                        'padding': '10px 15px',
                        'width': '100%'
                    }
                )
            ],
            None,
            False,
            [
                fac.AntdMessage(
                    content='回复生成成功',
                    type='success'
                ),
                # 新的滚动动作
                fuc.FefferyScroll(
                    scrollTargetId='latest-response-begin',
                    scrollMode='target',
                    executeScroll=True,
                    containerId='chat-records'
                )
            ],
            # 根据是否处于多轮对话模式选择返回的状态存储数据
            {
                'status': '开启' if enable_multi_round else '关闭',
                'history': [
                    *(
                        multi_round_store.get('history') or []
                    ),
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": response
                    },
                ]
            }
        ]

    # 若当前回调由清空记录按钮触发
    elif dash.ctx.triggered_id == 'clear-exists-records' and clear_records_trigger:

        return [
            [
                origin_children[0]
            ],
            None,
            False,
            fac.AntdMessage(
                content='已清空',
                type='success'
            ),
            {
                'status': '开启' if enable_multi_round else '关闭',
                'history': []
            }
        ]

    # 若当前回调由多轮对话状态切换开关触发
    elif dash.ctx.triggered_id == 'enable-multi-round':

        return [
            [
                origin_children[0]
            ],
            None,
            False,
            fac.AntdMessage(
                content=(
                    '已开启多轮对话模式'
                    if enable_multi_round
                    else '已关闭多轮对话模式'
                ),
                type='success'
            ),
            {
                'status': '开启' if enable_multi_round else '关闭',
                'history': []
            }
        ]

    return [
        dash.no_update,
        dash.no_update,
        False,
        None,
        dash.no_update
    ]


@app.callback(
    Output('download-history-qa-records', 'href'),
    Input('export-history-qa-records', 'nClicks'),
    State('multi-round-store', 'data'),
    prevent_initial_call=True
)
def export_history_qa_records(nClicks, history_qa_records):
    '''
    处理将当前全部对话记录导出为markdown文件的操作
    '''

    if nClicks and history_qa_records.get('history'):

        # 拼接历史QA记录
        return_md_str = ''

        for record in history_qa_records['history']:
            if record['role'] == 'user':
                return_md_str += '\n#### 问题：{}\n'.format(record['content'])

            else:
                return_md_str += '\n#### 回答：\n{}'.format(record['content'])
                
        filename="问答记录{}.md".format(
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
        output_file_path = os.path.join(
                        'caches',
                        'md',
                        filename
                    )
        # 打开文件以写入模式
        with open(output_file_path, 'w', encoding='utf-8') as md_file:
            # 将字符串写入文件
            md_file.write(return_md_str)
            
        download_href='/download?path={}&file={}'.format('md', filename)

        return download_href


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8055)
