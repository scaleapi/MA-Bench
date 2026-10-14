package catalog

import (
	"list"
	"encoding/yaml"
)

// Renders the input into the entity Datadog ingests. Contract is schema.cue.
// kind: repository, matching scaleapi/sai -- nothing here runs under its own
// name, so system/service would not fit.

#Organization:   "scaleapi"
#CatalogVersion: "0.0"

_repoSlug: "\(#Organization)/\(input.repoName)"

// team:/product: are what Datadog's facets read.
_tags: list.Concat([[
	"team:\(input.owningTeam)",
	"product:\(input.repoName)",
	"repo:\(_repoSlug)",
	"catalog-version:\(#CatalogVersion)",
], input.tags])

_links: list.Concat([[
	{
		name:     "\(input.repoName) GitHub Repository"
		type:     "repo"
		provider: "github"
		url:      "https://github.com/\(_repoSlug)"
	},
	{
		name:     "\(input.owningTeam) GitHub owning team"
		type:     "other"
		provider: "github"
		url:      "https://github.com/orgs/\(#Organization)/teams/\(input.owningTeam)"
	},
], input.links])

// Namespaced so a second renderer could join this package without colliding.
datadog: entity: {
	apiVersion: "v3"
	kind:       "repository"
	metadata: {
		name:        "github.com/\(_repoSlug)"
		displayName: input.displayName
		description: input.description
		owner:       input.owningTeam
		tags:        _tags
		links:       _links
	}
	spec: {
		lifecycle: input.lifecycle
		tier:      input.tier
	}
}

datadog: output: yaml.Marshal(datadog.entity)
