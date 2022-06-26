from KHUDoll_AIModels import get_converters
from AIServer import ServerSession

# load converters 
converters = get_converters()

# session execution
m_chatserver = ServerSession(converters)
m_chatserver.startSession()