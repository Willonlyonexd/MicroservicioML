# schema_recommendation.py
from graphql import GraphQLObjectType, GraphQLField, GraphQLString, GraphQLID, GraphQLFloat, GraphQLList, GraphQLNonNull, GraphQLInputObjectType, GraphQLInt

# Definir tipos para el sistema de recomendación
product_type = GraphQLObjectType(
    name='Product',
    fields=lambda: {
        'id': GraphQLField(GraphQLNonNull(GraphQLID)),
        'name': GraphQLField(GraphQLNonNull(GraphQLString)),
        'description': GraphQLField(GraphQLString),
        'price': GraphQLField(GraphQLFloat),
        'category': GraphQLField(category_type),
        'imageUrl': GraphQLField(GraphQLString),
    }
)

category_type = GraphQLObjectType(
    name='Category',
    fields=lambda: {
        'id': GraphQLField(GraphQLNonNull(GraphQLID)),
        'name': GraphQLField(GraphQLNonNull(GraphQLString)),
    }
)

recommendation_type = GraphQLObjectType(
    name='Recommendation',
    fields=lambda: {
        'product': GraphQLField(GraphQLNonNull(product_type)),
        'score': GraphQLField(GraphQLNonNull(GraphQLFloat)),
        'reason': GraphQLField(GraphQLString),
    }
)

# Input types
context_input_type = GraphQLInputObjectType(
    name='ContextInput',
    fields=lambda: {
        'timeOfDay': GraphQLField(GraphQLString),
        'dayOfWeek': GraphQLField(GraphQLInt),
        'partySize': GraphQLField(GraphQLInt),
    }
)

recommendation_filter_type = GraphQLInputObjectType(
    name='RecommendationFilter',
    fields=lambda: {
        'excludeProductIds': GraphQLField(GraphQLList(GraphQLNonNull(GraphQLID))),
        'categoryIds': GraphQLField(GraphQLList(GraphQLNonNull(GraphQLID))),
        'context': GraphQLField(context_input_type),
    }
)

# Query fields to be added to the main schema
recommendation_query_fields = {
    'personalRecommendations': GraphQLField(
        GraphQLNonNull(GraphQLList(GraphQLNonNull(recommendation_type))),
        args={
            'clientId': GraphQLArgument(GraphQLNonNull(GraphQLID)),
            'tenantId': GraphQLArgument(GraphQLNonNull(GraphQLID)),
            'limit': GraphQLArgument(GraphQLInt, default_value=5),
            'filter': GraphQLArgument(recommendation_filter_type),
        },
        # resolver will be attached in resolvers_recommendation.py
    ),
    'generalRecommendations': GraphQLField(
        GraphQLNonNull(GraphQLList(GraphQLNonNull(recommendation_type))),
        args={
            'tenantId': GraphQLArgument(GraphQLNonNull(GraphQLID)),
            'limit': GraphQLArgument(GraphQLInt, default_value=10),
            'filter': GraphQLArgument(recommendation_filter_type),
        },
        # resolver will be attached in resolvers_recommendation.py
    ),
}