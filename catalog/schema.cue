package catalog

// What a repository may declare. Edit entity.catalog.input.yaml, then run
// ./catalog/generate.sh -- entity.datadog.yaml is generated from this.

// What GitHub allows in a repository name, which is wider than a team slug:
// uppercase, underscores and periods are all legal and reach us unchanged.
#RepoName: =~"^[A-Za-z0-9._-]+$"
#TeamSlug: =~"^[a-z0-9]+(-[a-z0-9]+)*$"

#Lifecycle: "production" | "staging" | "development" | "deprecated"

// Closed set: =~"^tier[0-9]+$" would accept tier99.
#Tier: "tier1" | "tier2" | "tier3"

#Link: {
	name!:     string & !=""
	type!:     string & !=""
	provider!: string & !=""
	url!:      =~"^https?://"
}

#Input: {
	repoName!:    #RepoName
	owningTeam!:  #TeamSlug
	description!: string & !=""

	displayName: (string & !="") | *repoName
	lifecycle:   #Lifecycle | *"development"
	tier:        #Tier | *"tier3"

	// Appended after the generated links.
	links: [...#Link] | *[]

	// Written "key:value"; Datadog treats a tag with no colon as a bare word.
	tags: [...(string & !="")] | *[]
}

input: #Input
