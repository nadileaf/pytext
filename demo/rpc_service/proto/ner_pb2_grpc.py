# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import demo.rpc_service.proto.ner_pb2 as ner__pb2


class NERStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.TagTexts = channel.unary_unary(
        '/ner.NER/TagTexts',
        request_serializer=ner__pb2.TextBatch.SerializeToString,
        response_deserializer=ner__pb2.EntitiesBatch.FromString,
        )
    self.TagText = channel.unary_unary(
        '/ner.NER/TagText',
        request_serializer=ner__pb2.Text.SerializeToString,
        response_deserializer=ner__pb2.Entities.FromString,
        )


class NERServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def TagTexts(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def TagText(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_NERServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'TagTexts': grpc.unary_unary_rpc_method_handler(
          servicer.TagTexts,
          request_deserializer=ner__pb2.TextBatch.FromString,
          response_serializer=ner__pb2.EntitiesBatch.SerializeToString,
      ),
      'TagText': grpc.unary_unary_rpc_method_handler(
          servicer.TagText,
          request_deserializer=ner__pb2.Text.FromString,
          response_serializer=ner__pb2.Entities.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'ner.NER', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
