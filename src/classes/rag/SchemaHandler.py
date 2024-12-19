from typing import Literal, Type, List, Tuple


class SchemaHandler:
    """
    Handles schema definitions for knowledge graph validation, including entities, relations,
    and validation schemas specifically for smart contract reentrancy detection.
    """

    @staticmethod
    def get_validation_schema() -> List[Tuple[str, str, str]]:
        """
        Retrieve the validation schema defining valid triples in the knowledge graph for reentrancy detection.

        :return: A list of tuples representing valid (entity, relation, entity) triples.
        """
        return [
            # Smart Contract-specific triples
            ("SMART_CONTRACT", "CONTAINS", "FUNCTION"),
            ("SMART_CONTRACT", "DEPLOYS", "SMART_CONTRACT"),
            ("SMART_CONTRACT", "INTERACTS_WITH", "EXTERNAL_CONTRACT"),
            ("SMART_CONTRACT", "USES", "VARIABLE"),
            ("SMART_CONTRACT", "CALLS", "FUNCTION"),
            ("SMART_CONTRACT", "SUFFERED_FROM", "VULNERABILITY"),
            ("SMART_CONTRACT", "ASSOCIATED_WITH", "REENTRANCY_PATTERN"),

            # Function-related triples
            ("FUNCTION", "CALLS", "FUNCTION"),
            ("FUNCTION", "CALLS", "EXTERNAL_FUNCTION"),
            ("FUNCTION", "CONTAINS", "STATE_CHANGE"),
            ("FUNCTION", "READS", "VARIABLE"),
            ("FUNCTION", "WRITES", "VARIABLE"),
            ("FUNCTION", "USES", "REENTRANCY_PATTERN"),
            ("FUNCTION", "TRIGGERED_BY", "TRANSACTION"),

            # Vulnerability-related triples
            ("VULNERABILITY", "AFFECTS", "FUNCTION"),
            ("VULNERABILITY", "RELATED_TO", "REENTRANCY_PATTERN"),
            ("VULNERABILITY", "EXPLOITS", "STATE_CHANGE"),
            ("VULNERABILITY", "FOUND_IN", "SMART_CONTRACT"),

            # State Change-related triples
            ("STATE_CHANGE", "MODIFIES", "VARIABLE"),
            ("STATE_CHANGE", "LEADS_TO", "VULNERABILITY"),
            ("STATE_CHANGE", "TRIGGERED_BY", "CALL"),

            # Variable-related triples
            ("VARIABLE", "MODIFIED_BY", "FUNCTION"),
            ("VARIABLE", "READ_BY", "FUNCTION"),
            ("VARIABLE", "AFFECTED_BY", "STATE_CHANGE"),

            # Call-related triples
            ("CALL", "MAKES", "EXTERNAL_CALL"),
            ("CALL", "RETURNS", "VALUE"),
            ("CALL", "RESULTS_IN", "STATE_CHANGE"),

            # Reentrancy-related triples
            ("REENTRANCY_PATTERN", "IDENTIFIED_IN", "FUNCTION"),
            ("REENTRANCY_PATTERN", "LEADS_TO", "VULNERABILITY"),
            ("REENTRANCY_PATTERN", "EXPLOITED_BY", "CALL"),

            # Transaction-related triples
            ("TRANSACTION", "TRIGGERS", "FUNCTION"),
            ("TRANSACTION", "LEADS_TO", "STATE_CHANGE"),
            ("TRANSACTION", "RESULTS_IN", "VULNERABILITY"),
            ("TRANSACTION", "SENT_TO", "SMART_CONTRACT"),
        ]

    @staticmethod
    def get_entities() -> Type[str]:
        """
        Retrieve the list of possible entity types for the knowledge graph.

        :return: A Literal type representing the valid entity types.
        """
        return Literal[
            "SMART_CONTRACT", "FUNCTION", "EXTERNAL_FUNCTION", "VARIABLE",
            "STATE_CHANGE", "CALL", "EXTERNAL_CALL", "REENTRANCY_PATTERN",
            "VULNERABILITY", "TRANSACTION", "EXTERNAL_CONTRACT", "VALUE"
        ]

    @staticmethod
    def get_relations() -> Type[str]:
        """
        Retrieve the list of possible relation types for the knowledge graph.

        :return: A Literal type representing the valid relation types.
        """
        return Literal[
            "CONTAINS", "CALLS", "READS", "WRITES", "MODIFIES", "TRIGGERED_BY",
            "LEADS_TO", "RESULTS_IN", "AFFECTS", "RELATED_TO", "FOUND_IN",
            "IDENTIFIED_IN", "EXPLOITED_BY", "SUFFERED_FROM", "DEPLOYS",
            "INTERACTS_WITH", "USES", "MAKES", "RETURNS", "SENT_TO"
        ]
