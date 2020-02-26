import logging
import sys
sys.path.append('../../')
from concurrent import futures
import grpc
import demo.rpc_service.proto.ner_pb2 as ner_pb2
import demo.rpc_service.proto.ner_pb2_grpc as ner_pb2_grpc
from demo.rpc_service.config import PORT
from demo.rpc_service.predict import Predictor


class NerServicer(ner_pb2_grpc.NERServicer):
    def __init__(self):
        self.predictor = Predictor()

    def TagTexts(self, request, context):
        texts = request.texts
        entities_batch = []
        for text in texts:
            res = self.predictor.predict(text)
            entities = []
            for ent in res:
                entities.append(ner_pb2.Entity(value=ent['value'],
                                               start=ent['start'],
                                               end=ent['end'],
                                               entity=ent['entity']))
            entities_batch.append(ner_pb2.Entities(entities=entities))
        return ner_pb2.EntitiesBatch(
            entities_batch=entities_batch
        )

    def TagText(self, request, context):
        text = request.text
        res = self.predictor.predict(text)
        entities = []
        for ent in res:
            entities.append(ner_pb2.Entity(value=ent['value'],
                                             start=ent['start'],
                                             end=ent['end'],
                                             entity=ent['entity']))
        return ner_pb2.Entities(
            entities=entities
        )


def serve():
    import time
    SECONDAS_IN_DAY = 60 * 60 * 24
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)])
    ner_pb2_grpc.add_NERServicer_to_server(NerServicer(), server)
    server.add_insecure_port(f'[::]:{PORT}')
    server.start()
    try:
        while True:
            time.sleep(SECONDAS_IN_DAY)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
