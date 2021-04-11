import msgpack

def pack_req(req_id, text):
    return msgpack.packb((req_id, text), use_bin_type=True)

def unpack_req(req_bin):
    return msgpack.unpackb(req_bin, use_list=False, raw=False)